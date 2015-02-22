%%
clear; clc;
%% training data
dir_data = 'C:\Dev\code\matconvnet\examples\data\mnist_small_cv5';
fn_data = fullfile(dir_data, 'imdb.mat');
load(fn_data);

% draw a batch
batch_sz = 128;
ind_tr = find(images.set==1);
ind_bat = ind_tr( randsample(numel(ind_tr), batch_sz) );
X = images.data(:,:,:, ind_bat);
Y = images.labels(:, ind_bat);
%% the network
f = 1/100;
% 1: conv, param
h = tf_conv(); 
h.i = n_data(); 
h.o = n_data();
h.p = [n_data(), n_data()];
h.p(1).a = f*randn(5,5,1,20, 'single'); % kernel
h.p(2).a = zeros(1, 20, 'single'); % bias
tfs{1} = h;
% 2: pool
h = tf_pool();
h.i = tfs{1}.o; 
h.o = n_data();
tfs{2} = h;
% 3: conv, param
h = tf_conv();
h.i = tfs{2}.o;
h.o = n_data();
h.p = [n_data(), n_data()];
h.p(1).a = f*randn(5,5,20,50, 'single');
h.p(2).a = zeros(1,50,'single');
tfs{3} = h;
% 4: pool
h = tf_pool();
h.i = tfs{3}.o;
h.o = n_data();
tfs{4} = h;
% 5: full connection, param
h = tf_conv();
h.i = tfs{4}.o;
h.o = n_data();
h.p = [n_data(), n_data()];
h.p(1).a = f*randn(4,4,50,500, 'single');
h.p(2).a = zeros(1,500,'single');
tfs{5} = h;
% 6: full connection, param
h = tf_conv();
h.i = tfs{5}.o;
h.o = n_data();
h.p = [n_data(), n_data()];
h.p(1).a = f*randn(1,1,500,10, 'single');
h.p(2).a = zeros(1,10,'single');
tfs{6} = h;
% 7: loss
h = tf_loss_lse();
h.i = [tfs{6}.o, n_data()];
h.o = n_data();
tfs{7} = h;
%% the parameters
% collect the parameters
params = [tfs{1}.p(:); tfs{3}.p(:); tfs{5}.p(:); tfs{6}.p(:)];
% create the corresponding numeric optimizers
opt_arr(numel(params), 1) = opt_1storder();
%% do the training
% set the source/root & sink/leaf data node
tfs{1}.i.a    = X; %
tfs{7}.i(2).a = Y; % 
tfs{7}.o.d    = 1;

% fprop & bprop
t_elapse = tic;
tfs           = cellfun(@fprop, tfs,           'uniformoutput',false) ;
tfs(end:-1:1) = cellfun(@bprop, tfs(end:-1:1), 'uniformoutput',false) ;
t_elapse = toc(t_elapse);
fprintf('batch time = %d\n', t_elapse);

% update parameters
opt_arr = arrayfun(@update, opt_arr, params, 'uniformoutput',false);