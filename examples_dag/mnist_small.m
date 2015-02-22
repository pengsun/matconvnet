function [net, info] = mnist_small(varargin)
% mnist_small  Demonstrate dagcnn_train.m on dataset mnist_small
%    Adopted from "./exmaples/cnn_mnist"

%%% config
dir_root = vl_rootnn();
% output model
opts.expDir  = fullfile(dir_root, '/matlab_dag/mo_zoo','mnist_small_cv5') ;
% input data
opts.dataDir  = fullfile(dir_root, 'data','mnist_small_cv5') ;
% opts = vl_argparse(opts, varargin) ;
opts.imdbPath = fullfile(opts.dataDir, 'imdb.mat');
% cv partition
opts.cvk = 1;
% training 
opts.train.batchSize = 128 ;
opts.train.numEpochs = 25 ;
opts.train.continue = true ;
opts.train.useGpu = false ;
opts.train.learningRate = 0.001 ;
opts.train.expDir = opts.expDir ;
opts = vl_argparse(opts, varargin) ;

%%% load data
if exist(opts.imdbPath, 'file')
  imdb = load(opts.imdbPath) ;
else
  error('cannnot find data %s\n', opts.imdbPath );
end

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

%%% network layers
f = 1/100 ;
net.layers = {} ;
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(5,5,1,20, 'single'), ...
                           'biases', zeros(1, 20, 'single'), ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(5,5,20,50, 'single'),...
                           'biases', zeros(1,50,'single'), ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(4,4,50,500, 'single'),...
                           'biases', zeros(1,500,'single'), ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'relu') ;
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(1,1,500,10, 'single'),...
                           'biases', zeros(1,10,'single'), ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'lse') ;
% net.layers{end+1} = struct('type', 'softmaxloss') ;
% imdb.images.labels = imdb.images.labels_old;


%%% train with SGD

% Take the mean out (where?) and make GPU if needed
if opts.train.useGpu
  imdb.images.data = gpuArray(imdb.images.data) ;
end

[net, info] = cnn_train(net, imdb, @getBatch, ...
    opts.train) ;


function [im, labels] = getBatch(imdb, batch)
im = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(:,batch) ;

function initNet

