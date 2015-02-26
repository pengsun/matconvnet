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
h = tf_conv(); 
h.p(1).a = f*randn(5,5,1,20, 'single'); % kernel
h.p(2).a = zeros(1, 20, 'single'); % bias
tfs{1} = h;
% 2: pool
h = tf_pool();
h.i = tfs{1}.o; 
tfs{2} = h;
% 3: conv, param
h = tf_conv();
h.i = tfs{2}.o;
h.p(1).a = f*randn(5,5,20,50, 'single');
h.p(2).a = zeros(1,50,'single');
tfs{3} = h;
% 4: pool
h = tf_pool();
h.i = tfs{3}.o;
tfs{4} = h;
% 5: full connection, param
h = tf_conv();
h.i = tfs{4}.o;
h.p(1).a = f*randn(4,4,50,500, 'single');
h.p(2).a = zeros(1,500,'single');
tfs{5} = h;
% 6: relu
h = tf_relu();
h.i = tfs{5}.o;
tfs{6} = h;
% 7: full connection, param
h = tf_conv();
h.i = tfs{6}.o;
h.p(1).a = f*randn(1,1,500,10, 'single');
h.p(2).a = zeros(1,10,'single');
tfs{7} = h;
% 8: loss
h = tf_loss_lse();
h.i(1) = tfs{7}.o;
tfs{8} = h;
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