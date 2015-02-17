function info = mnist_small_cv5_test_as_val(varargin)
% mnist_small_cv5_evaluate   Evauate MatConvNet models on ImageNet

%%% config
run(fullfile(fileparts(mfilename('fullpath')), ...
  '..', 'matlab', 'vl_setupnn.m')) ;
opts.dataDir = fullfile('data', 'mnist_small_cv5') ;
opts.expDir = fullfile('exp', 'mnist_small_cv5') ;
opts.imdbPath = fullfile(opts.dataDir, 'imdb.mat');
opts.modelPath = fullfile('mo_zoo', 'mnist_small_cv5', 'net-epoch-16.mat') ;

opts.train.batchSize = 128 ;
opts.train.numEpochs = 1 ;
opts.train.useGpu = false ;
opts.train.prefetch = false ;
opts.train.expDir = opts.expDir ;

opts = vl_argparse(opts, varargin) ;
display(opts);

%%% load data
imdb = load(opts.imdbPath) ;

%%% load model
tmp = load(opts.modelPath) ;
net = tmp.net; clear tmp;
% net.layers = net.layers(1:end-1);
% net.layers{end}.type = 'softmaxloss' ; % softmax -> softmaxloss


% -------------------------------------------------------------------------
%                                               Stochastic gradient descent
% -------------------------------------------------------------------------

[net,info] = cnn_train(net, imdb, @getBatch, opts.train, ...
  'conserveMemory', true, ...
  'train', NaN, ...
  'val', find(imdb.images.set==3)) ; % trick: testing as validation

%%% print
tmp = load(opts.modelPath) ; info_tr = tmp.info; clear tmp;
fprintf('tr err = %d\n', info_tr.train.error(end) );
fprintf('te err = %d\n', info.val.error);

function [im, labels] = getBatch(imdb, batch)
im = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(:,batch) ;
