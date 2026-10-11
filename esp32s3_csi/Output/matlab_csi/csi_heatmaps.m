function results = csi_heatmaps(userFiles, referenceSource, outputDir, windowSeconds, binSeconds)
% CSI_HEATMAPS Real CSI amplitude heatmaps and descriptive comparisons.
% MATLAB R2020b+; no Statistics or Signal Processing Toolbox required.
% CSV complex pairs must be [imaginary, real]. No eval, interpolation or synthesis.
% Example: csi_heatmaps(userFiles, 'csi_data.zip', 'results', 60, 0.1)
if nargin < 4, windowSeconds = 60; end
if nargin < 5, binSeconds = 0.1; end
validateattributes(windowSeconds, {'numeric'}, {'scalar','positive','finite'});
validateattributes(binSeconds, {'numeric'}, {'scalar','positive','finite','<=',windowSeconds});
if nargin < 3 || isempty(outputDir), outputDir = fullfile(pwd,'csi_figures'); end
if ~isfolder(outputDir), mkdir(outputDir); end
userFiles = string(userFiles(:));
referenceSource = string(referenceSource);
if endsWith(referenceSource,'.zip','IgnoreCase',true)
    unpacked = fullfile(outputDir,'reference_extracted');
    if ~isfolder(unpacked), mkdir(unpacked); end
    unzip(referenceSource, unpacked);
    sourceDir = unpacked;
else
    sourceDir = referenceSource;
end
refs = dir(fullfile(sourceDir,'**','*.csv'));
referenceFiles = strings(numel(refs),1);
for i=1:numel(refs), referenceFiles(i)=fullfile(refs(i).folder,refs(i).name); end
referenceFiles = sort(referenceFiles);
files = [userFiles; referenceFiles];
assert(~isempty(files),'No CSV files found.');
results = cell(numel(files),1);
for i=1:numel(files)
    fprintf('[%d/%d] %s\n',i,numel(files),files(i));
    results{i} = readCSI(files(i),windowSeconds,binSeconds);
    if i<=numel(userFiles)
        results{i}.group = "Measured";
        results{i}.name = "Measured " + string(i);
    else
        [parent,~,~] = fileparts(files(i));
        [~,folderName,~] = fileparts(parent);
        results{i}.group = "Reference";
        results{i}.name = "Reference " + string(folderName);
    end
end

% Shared limits over all plotted cells. Clipping is disclosed in titles/README.
amps = []; deltas = [];
for i=1:numel(results)
    r=results{i};
    amps=[amps; r.amplitude(isfinite(r.amplitude))]; %#ok<AGROW>
    deltas=[deltas; abs(r.relative(isfinite(r.relative)))]; %#ok<AGROW>
end
ampLimit = max(1,localPercentile(amps,99));
deltaLimit = max(1,localPercentile(deltas,99));
summary=table;
for i=1:numel(results)
    r=results{i};
    summary=[summary; table(r.group,r.name,string(r.file),r.rowsTotal,r.framesUsed,...
        r.duration,r.fps,r.badRows,r.otherLengths,r.seqGaps,r.emptyBins,...
        r.length,r.channel,r.bandwidth,r.sigMode,r.rssi,r.medianCV,...
        'VariableNames',{'Group','Name','File','TotalRows','FramesInWindow',...
        'RecordedSpanSeconds','ObservedFPS','BadRowsInWindow','OtherLengthsInWindow',...
        'SequenceGapsInWindow','EmptyTimeBins','CSIBytes','Channel','CWB','SigMode',...
        'MedianRSSI','MedianSubcarrierCV'})]; %#ok<AGROW>
end
writetable(summary,fullfile(outputDir,'summary.csv'));
config=struct('windowSeconds',windowSeconds,'binSeconds',binSeconds,...
    'amplitudeLimits',[0 ampLimit],'relativePercentLimits',[-deltaLimit deltaLimit],...
    'aggregation','median per equal-duration bin; gaps remain NaN',...
    'relativeDefinition','100*(bin amplitude / per-pair window median - 1)',...
    'comparisonWarning','Reference LTF/subcarrier mapping unknown. Bandwidth/channel/gain differ. Descriptive comparison only.');
fid=fopen(fullfile(outputDir,'plot_settings.json'),'w');
assert(fid>=0,'Cannot write plot settings.');
closer=onCleanup(@()fclose(fid));
fprintf(fid,'%s',jsonencode(config));
clear closer;

for i=1:numel(results)
    r=results{i};
    f=figure('Visible','off','Color','w','Position',[50 50 1200 760]);
    lay=tiledlayout(f,2,1,'TileSpacing','compact','Padding','compact');
    drawPanel(nexttile(lay),r,"amplitude",ampLimit,windowSeconds);
    drawPanel(nexttile(lay),r,"relative",deltaLimit,windowSeconds);
    title(lay,r.name+" | "+string(r.framesUsed)+" frames in plotted window",'Interpreter','none');
    exportgraphics(f,fullfile(outputDir,sprintf('%02d_heatmap.png',i)),'Resolution',180);
    savefig(f,fullfile(outputDir,sprintf('%02d_heatmap.fig',i)));
    close(f);
end
drawOverview(results,"amplitude",ampLimit,windowSeconds,outputDir);
drawOverview(results,"relative",deltaLimit,windowSeconds,outputDir);
save(fullfile(outputDir,'processed_csi.mat'),'results','summary','config','-v7.3');
fprintf('Saved real-data figures and summary to: %s\n',outputDir);
end

function r=readCSI(file,windowSeconds,binSeconds)
opts=detectImportOptions(file,'Delimiter',',','VariableNamingRule','preserve');
opts=setvartype(opts,opts.VariableNames,'string');
T=readtable(file,opts);
required={'data','len','first_word_invalid'};
assert(all(ismember(required,T.Properties.VariableNames)),...
    'CSV is missing data, len or first_word_invalid: %s',file);
if ismember('type',T.Properties.VariableNames)
    T=T(T.type=="CSI_DATA" | T.type=="CSI",:);
end
assert(height(T)>1,'At least two data rows are needed: %s',file);
[t,timeSource]=getTime(T);
goodTime=isfinite(t);
T=T(goodTime,:); t=t(goodTime);
assert(numel(t)>1 && all(diff(t)>=0),'Time reset or invalid timestamps in %s',file);
t=t-t(1);
r.file=char(file); r.rowsTotal=height(T); r.duration=t(end);
r.fps=(numel(t)-1)/max(eps,t(end)); r.timeSource=timeSource;
keep=t<windowSeconds;
T=T(keep,:); t=t(keep);
L=str2double(T.len);
validLen=isfinite(L) & L>0 & mod(L,2)==0;
assert(any(validLen),'No even positive CSI lengths in %s',file);
r.length=mode(L(validLen));
r.otherLengths=sum(~validLen | L~=r.length);
keep=validLen & L==r.length;
T=T(keep,:); t=t(keep);
A=nan(height(T),r.length/2);
ok=false(height(T),1);
for i=1:height(T)
    try
        v=jsondecode(char(T.data(i)));
        invalid=str2double(T.first_word_invalid(i));
        if ~isnumeric(v) || ~isvector(v) || numel(v)~=r.length || ...
                any(~isfinite(v) | fix(v)~=v | v<-128 | v>127) || ...
                ~ismember(invalid,[0 1])
            continue;
        end
        v=double(v(:));
        A(i,:)=hypot(v(1:2:end),v(2:2:end));
        if invalid==1, A(i,1:min(2,size(A,2)))=NaN; end
        ok(i)=true;
    catch
        % Malformed arrays are counted and excluded; never evaluated as code.
    end
end
r.badRows=sum(~ok);
T=T(ok,:); t=t(ok); A=A(ok,:);
assert(size(A,1)>1,'Not enough valid CSI frames: %s',file);
r.framesUsed=size(A,1);
r.seqGaps=NaN;
if ismember('seq',T.Properties.VariableNames)
    sequence=str2double(T.seq);
    if all(isfinite(sequence))
        assert(all(diff(sequence)>0),'Sequence reset/duplicates in %s',file);
        r.seqGaps=sum(max(0,diff(sequence)-1));
    end
end
r.channel=fieldLevels(T,{'channel_primary','channel'});
r.bandwidth=fieldLevels(T,{'cwb','bandwidth'});
r.sigMode=fieldLevels(T,{'sig_mode'});
r.rssi=fieldMedian(T,'rssi');
r.labels=fieldLevels(T,{'taget','label'});
r.baseline=median(A,1,'omitnan');
means=mean(A,1,'omitnan');
cv=std(A,0,1,'omitnan')./means;
cv(means<=0)=NaN;
r.medianCV=median(cv,'omitnan');
edges=0:binSeconds:windowSeconds;
if edges(end)<windowSeconds, edges(end+1)=windowSeconds; end
assert(max(abs(diff(edges)-binSeconds))<1e-8,...
    'windowSeconds must be an integer multiple of binSeconds.');
bins=discretize(t,edges);
nb=numel(edges)-1;
B=nan(nb,size(A,2));
counts=zeros(nb,1);
for b=1:nb
    rows=bins==b;
    counts(b)=sum(rows);
    if any(rows), B(b,:)=median(A(rows,:),1,'omitnan'); end
end
r.emptyBins=sum(counts==0);
r.binCounts=counts;
r.time=(edges(1:end-1)+edges(2:end))/2;
r.amplitude=B';
denom=r.baseline; denom(denom<=0)=NaN;
r.relative=(100*(B./denom-1))';
r.pairIndices=1:size(A,2);
end

function [t,source]=getTime(T)
vars=T.Properties.VariableNames;
if ismember('local_timestamp',vars), key='local_timestamp';
elseif ismember('esp_timestamp_us',vars), key='esp_timestamp_us';
else, key=''; end
if ~isempty(key)
    raw=str2double(T.(key));
    if all(isfinite(raw))
        d=diff(raw);
        assert(~any(d<0 & d>-2^31),'Device timestamp reset (not 32-bit wrap).');
        wraps=[0;cumsum(d < -2^31)];
        t=(raw+wraps*2^32-raw(1))/1e6;
        source="device microseconds"; return;
    end
end
if ismember('host_elapsed_s',vars)
    t=str2double(T.host_elapsed_s); source="PC elapsed seconds"; return;
end
assert(ismember('timestamp',vars),'No timestamp column available.');
dt=datetime(T.timestamp,'InputFormat','yyyy-MM-dd HH:mm:ss.SSS');
t=seconds(dt-dt(1)); source="PC wall timestamp";
end

function v=fieldLevels(T,names)
v="unavailable";
for i=1:numel(names)
    if ismember(names{i},T.Properties.VariableNames)
        s=unique(T.(names{i})); s=s(~ismissing(s));
        v=strjoin(s,';'); return;
    end
end
end

function v=fieldMedian(T,name)
v=NaN;
if ismember(name,T.Properties.VariableNames)
    v=median(str2double(T.(name)),'omitnan');
end
end

function q=localPercentile(v,p)
v=sort(v(isfinite(v))); assert(~isempty(v),'No valid heatmap cells.');
at=1+(numel(v)-1)*p/100;
lo=floor(at); hi=ceil(at);
q=v(lo)+(at-lo)*(v(hi)-v(lo));
end

function drawPanel(ax,r,kind,limit,windowSeconds)
if kind=="amplitude"
    Z=r.amplitude; limits=[0 limit]; palette=parula(256);
    caption="Amplitude (driver units)";
else
    Z=r.relative; limits=[-limit limit];
    palette=interp1([0 .5 1],[.13 .32 .65; 1 1 1; .8 .32 .08],linspace(0,1,256));
    caption="Change from own window median (%)";
end
im=imagesc(ax,r.time,r.pairIndices,Z);
im.AlphaData=isfinite(Z);
ax.Color=[.94 .94 .94];
set(ax,'YDir','normal','FontSize',10,'Box','on');
xlim(ax,[0 windowSeconds]);
ylim(ax,[.5 numel(r.pairIndices)+.5]);
colormap(ax,palette); caxis(ax,limits);
cb=colorbar(ax); cb.Label.String=caption;
xlabel(ax,'Elapsed time (s)');
ylabel(ax,'Stored complex-pair index');
title(ax,r.name+" | ch="+r.channel+", cwb="+r.bandwidth+", n="+string(r.framesUsed),...
    'Interpreter','none','FontSize',11);
end

function drawOverview(results,kind,limit,windowSeconds,outputDir)
N=numel(results);
f=figure('Visible','off','Color','w','Position',[20 20 1650 max(620,290*ceil(N/2))]);
lay=tiledlayout(f,ceil(N/2),2,'TileSpacing','compact','Padding','compact');
for i=1:N
    drawPanel(nexttile(lay),results{i},kind,limit,windowSeconds);
end
title(lay,"CSI "+kind+" | shared scale (99th percentile clipping); "+...
    "reference mapping unknown",'Interpreter','none','FontSize',15);
exportgraphics(f,fullfile(outputDir,"comparison_"+kind+".png"),'Resolution',160);
savefig(f,fullfile(outputDir,"comparison_"+kind+".fig"));
close(f);
end
