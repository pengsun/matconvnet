%%
clear; clc;
%% training data
dir_data = fullfile(vl_rootnn,'\examples\data\mnist_small_cv5');
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
tfs{1}        = tf_conv(); 
tfs{1}.p(1).a = f*randn(5,5,1,20, 'single'); % kernel
tfs{1}.p(2).a = zeros(1, 20, 'single'); % bias
% 2: pool
tfs{2}   = tf_pool();
tfs{2}.i = tfs{1}.o; 
% 3: conv, param
tfs{3}        = tf_conv();
tfs{3}.i      = tfs{2}.o;
tfs{3}.p(1).a = f*randn(5,5,20,50, 'single');
tfs{3}.p(2).a = zeros(1,50,'single');
% 4: pool
tfs{4}   = tf_pool();
tfs{4}.i = tfs{3}.o;
% 5: full connection, param
tfs{5}        = tf_conv();
tfs{5}.i      = tfs{4}.o;
tfs{5}.p(1).a = f*randn(4,4,50,500, 'single');
tfs{5}.p(2).a = zeros(1,500,'single');
% 6: relu
tfs{6}   = tf_relu();
tfs{6}.i = tfs{5}.o;
% 7: full connection, param
tfs{7}        = tf_conv();
tfs{7}.i      = tfs{6}.o;
tfs{7}.p(1).a = f*randn(1,1,500,10, 'single');
tfs{7}.p(2).a = zeros(1,10,'single');
% 8: loss
tfs{8}      = tf_loss_lse();
tfs{8}.i(1) = tfs{7}.o;
%% the parameters
% collect the parameters
params = [tfs{1}.p(:); tfs{3}.p(:); tfs{5}.p(:); tfs{7}.p(:)];
% create the corresponding numeric optimizers
opt_arr(numel(params), 1) = opt_1storder();
%% do the training
% set the source/root & sink/leaf data node
tfs{1}.i.a    = X; %
tfs{8}.i(2).a = Y; % 
tfs{8}.o.d    = 1;

% fprop & bprop
t_elapse = tic;
tfs           = cellfun(@fprop, tfs,           'uniformoutput',false) ;
tfs(end:-1:1) = cellfun(@bprop, tfs(end:-1:1), 'uniformoutput',false) ;
t_elapse = toc(t_elapse);
fprintf('batch time = %d\n', t_elapse);

% update parameters
for i = 1 : numel(opt_arr)
  opt_arr(i) = update(opt_arr(i), params(i));
end