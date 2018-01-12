dirname = '/Volumes/H.H. Lab (fab)/Fabio/data/lab/eb3';
files=dir(dirname);
dirFlags = [files.isdir];
subFolders = files(dirFlags);
for k = 1 : length(subFolders)    
    % folders inside this directory are conditions
    if ~strcmp(subFolders(k).name, '.') && ~strcmp(subFolders(k).name, '..')
      	fprintf('Sub folder #%d = %s\n', k, subFolders(k).name);
        condition = subFolders(k).name;
        dircond = [dirname, '/', condition];
        files = dir(dircond);
        dateFolders = files([files.isdir]);
        
        % folders inside this directory are dates        
        for l = 1 : length(dateFolders)
            if ~strcmp(dateFolders(l).name, '.') && ~strcmp(dateFolders(l).name, '..')
                fprintf('Date sub folder #%d = %s\n', l, dateFolders(l).name);
                date = dateFolders(l).name;
                dirdate = [dircond, '/', date];
                files = dir(dirdate);
                resultFolders = files([files.isdir]);

                % folders inside this directory are results        
                for m = 1 : length(resultFolders)
                    if ~strcmp(resultFolders(m).name, '.') && ~strcmp(resultFolders(m).name, '..')
                        fprintf('Result sub folder #%d = %s\n', m, resultFolders(m).name);
                        result = resultFolders(m).name;
                        dirres = [dirdate, '/', result];
                        file = dir([dirres, '/*.mat']);

                        for n = 1 : length(file)
                            if ~strcmp(file(n).name, 'time.mat')
                                load([dirres, '/', file.name])
                                timeInterval = struct(MD).timeInterval_;
                                tfile = [dirres, '/time.mat'];
                                save(tfile,'timeInterval')
                            end
                        end
                    end
                end
            end
        end
    end       
end