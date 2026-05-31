function compute_iqa_matlab(targetDir, metricName, saveFile)
% compute_iqa_matlab Measure NIQE or BRISQUE scores in MATLAB.
%   compute_iqa_matlab(targetDir, metricName, saveFile)
%   - targetDir : root folder of target images
%   - metricName: 'niqe' or 'brisque' (case insensitive)
%   - saveFile  : CSV save path (e.g., 'results/niqe.csv')
%
% Steps:
%   1) read 'GyroVD_Real.txt' to get Day/Video or Video folder list
%   2) collect all *.png images under targetDir matching those paths
%   3) resize images to 1080x1920 -> 540x960 (bicubic), same as the python code
%   4) compute NIQE or BRISQUE
%   5) write individual scores and overall average to CSV
    
    arguments
        targetDir (1,1) string
        metricName (1,1) string {mustBeMemberi(metricName, ["niqe","brisque"])}
        saveFile (1,1) string
    end
        
    disp(saveFile);
    
    % resize config
    newWidth  = 1080/2;  % 540
    newHeight = 1920/2;  % 960

    % read video list text file
    listFile = "GyroVD_Real.txt";
    if ~isfile(listFile)
        error("'%s' not found in current directory.", listFile);
    end
    lines = strtrim(splitlines(string(fileread(listFile))));
    lines = lines(lines ~= "");
        
    % collect image paths
    imgPaths = strings(0,1);

    for i = 1:numel(lines)
        parts = split(lines(i), "/");
        if numel(parts) >= 2
            dayName   = strtrim(parts(1));
            videoName = strtrim(parts(2));
        else
            dayName   = "";
            videoName = strtrim(parts(1));
        end

        % search patterns
        patterns = [
            fullfile(targetDir, dayName, videoName, '*.png')
            fullfile(targetDir, videoName, '*.png')
            fullfile(targetDir, dayName, 'IMUDVD-real', videoName, '*.png')
            fullfile(targetDir, 'IMUDVD-real', dayName, videoName, '*.png')
        ];

        files = dir(patterns(1));
        if isempty(files), files = dir(patterns(2)); end
        if isempty(files), files = dir(patterns(3)); end
        if isempty(files), files = dir(patterns(4)); end

        if ~isempty(files)
            [~, idx] = sort({files.name});
            files = files(idx);

            fulls = strings(numel(files),1);
            for k = 1:numel(files)
                fulls(k) = string(fullfile(files(k).folder, files(k).name));
            end

            imgPaths = [imgPaths; fulls];
        end
    end

    fprintf('Total images: %d\n', numel(imgPaths));

    % prepare CSV
    outDir = fileparts(saveFile);
    if outDir ~= "" && ~isfolder(outDir)
        mkdir(outDir);
    end

    fid = fopen(saveFile, 'w');
    if fid == -1
        error("Cannot open CSV file: %s", saveFile);
    end
    cleanupObj = onCleanup(@() fclose(fid));

    % metric mode
    useNIQE    = strcmpi(metricName, "niqe");
    useBRISQUE = strcmpi(metricName, "brisque");

    scores = zeros(numel(imgPaths), 1);

    tStart = tic;
    for i = 1:numel(imgPaths)
        imgPath = imgPaths(i);
        
        try
            I = imread(imgPath);
        catch
            disp(imgPath);
            error('Cannot read image.');
        end

        % resize bicubic to 540x960 (H,W)
        I = imresize(I, [newHeight, newWidth], 'bicubic');

        % compute metric
        if useNIQE
            score = niqe(I);
        elseif useBRISQUE
            score = brisque(I);
        else
            error('Invalid metricName: %s', metricName);
        end
        
        scores(i) = score;

        % write per-image row: "video/file,score"
        parentDir = string(fileparts(imgPath));
        [~, videoFolder] = fileparts(parentDir);

        [~, fileName, ext] = fileparts(imgPath);
        relOut = sprintf('%s/%s', videoFolder, strcat(fileName, ext));
        fprintf(fid, "%s,%.6f\n", relOut, score);

        if mod(i, 200) == 0 || i == numel(imgPaths)
            fprintf('Processed %d / %d (elapsed: %.1fs)\n', ...
                i, numel(imgPaths), toc(tStart));
        end
    end

    % overall average
    avgScore = mean(scores);
    fprintf(fid, "AVG,%.6f\n", avgScore);

    fprintf('Average %s: %.6f (n=%d)\n', ...
        upper(metricName), avgScore, numel(imgPaths));
    fprintf('Done! Results are in %s\n', saveFile);
end

function mustBeMemberi(val, choices)
    tf = any(strcmpi(val, choices));
    if ~tf
        throwAsCaller(MException('validate:mustBeMemberi', ...
            'Value must be one of: %s', strjoin(choices, ', ')));
    end
end