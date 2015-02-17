function info = mnist_small_cv5_test(varargin)
% CNN_MNIST  Demonstrated MatConNet on MNIST

%%% config
run(fullfile(fileparts(mfilename('fullpath')),...
  '..', 'matlab', 'vl_setupnn.m')) ;
% model
opts.moDir  = fullfile('mo_zoo','mnist_small_cv5') ;
opts.moPath = fullfile(opts.moDir, 'net-epoch-16.mat');
% data
opts.dataDir  = fullfile('data','mnist_small_cv5') ;
opts.imdbPath = fullfile(opts.dataDir, 'imdb.mat');
% 
opts.batchSize = 17;
% cv partition
opts.cvk = 1;
opts = vl_argparse(opts, varargin) ;

%%%
disp(opts);

%%% load data
imdb = load(opts.imdbPath) ;

%%% set working set
switch opts.cvk
  case 1
    imdb.images.set = imdb.images.set1;
  case 2
    imdb.images.set = imdb.images.set2;
  case 3
    imdb.images.set = imdb.images.set3;
  case 4
    imdb.images.set = imdb.images.set4;
  case 5
    imdb.images.set = imdb.images.set5;
  otherwise
    error('opts.cvk: wrong number\n');
end

%%% load network layers
tmp = load( opts.moPath );
net = tmp.net;
net.layers = net.layers(1:end-1); % remove the loss layer at tail
clear tmp;
% fprintf('%s loaded...\n', opts.moPath);


% % Take the mean out (where?) and make GPU if needed
% if opts.train.useGpu
%   imdb.images.data = gpuArray(imdb.images.data) ;
% end

%%% Test
ind_te = find( imdb.images.set == 3 );
for t = 1 : opts.batchSize : numel(ind_te)
  % get the batch
  iend = min(t+opts.batchSize-1, numel(ind_te));
  batch = ind_te( t : iend);
  X = imdb.images.data(:,:,1,batch);
  
  % test the batch
  batch_time = tic;
  [tmp,~] = cnn_test_batch(net, X);
  batch_time = toc(batch_time);
  
  % print information
  fprintf('testing batch %d of %d...',...
    ceil(t/opts.batchSize), ceil(numel(ind_te)/opts.batchSize) );
  speed = numel(batch)/batch_time;
  fprintf(' %.2f s (%.1f images/s) \n', batch_time, speed) ;
  
  % get the restuls
  if (t == 1), Y = tmp;
  else Y = [Y, tmp];  %#ok<AGROW>
  end % if
end % for

%%% classification error rate
[~,iY] = max(Y);
gt = imdb.images.labels(:, ind_te);
[~,igt] = max(gt);
err = sum( iY ~= igt );
err = err/numel(ind_te);
fprintf('cls err = %d', err);

%%% wrap up the results
info.Y = Y;
info.err = err;
