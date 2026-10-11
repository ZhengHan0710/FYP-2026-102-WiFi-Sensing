% Open this script in MATLAB and click Run. All source files are read-only.
scriptDir=fileparts(mfilename('fullpath'));
addpath(scriptDir);
userDir='D:\ESP32\program\esp32s3_csi\data\20261011_190657_739143';
referenceZip='C:\Users\16265\xwechat_files\wxid_l5j7jbboxfon22_141e\msg\file\2026-10\csi_data.zip';
if isfolder(userDir)
    entries=dir(fullfile(userDir,'*.csv'));
    userFiles=sort(string(fullfile({entries.folder},{entries.name})));
else
    [names,folder]=uigetfile('*.csv','Select your three measured CSV files','MultiSelect','on');
    assert(~isequal(names,0),'No measured files selected.');
    userFiles=string(fullfile(folder,cellstr(string(names))));
end
assert(numel(userFiles)==3,'Select exactly three measured CSV files.');
if ~isfile(referenceZip)
    [name,folder]=uigetfile('*.zip','Select csi_data.zip');
    assert(~isequal(name,0),'No reference archive selected.');
    referenceZip=fullfile(folder,name);
end
outputDir=fullfile(scriptDir,'results');
results=csi_heatmaps(userFiles,referenceZip,outputDir,60,0.1);
fprintf('Open comparison_amplitude.png and comparison_relative.png in %s\n',outputDir);
