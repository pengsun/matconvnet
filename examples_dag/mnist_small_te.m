function mnist_small_te()
%% put all the stuff in a static method of convdag_lenet if you like
%% init dag: from file 
fn_mo = 'dag_epoch_10.mat';
dir_mo = 'D:\CodeWork\git\psmatconvnet\examples_dag\mo_zoo\mnist_small\lenet';
ffn_mo = fullfile(dir_mo, fn_mo);
load(ffn_mo, 'ob');
% get ob from here
%% config 
% TODO: add more properties here
ob.batch_sz = 128;
fn_data = 'D:\CodeWork\git\psmatconvnet\examples\data\mnist_small_cv5\imdb.mat';
%% do the training
[X, Y] = load_te_data(fn_data);
Ypre = test(ob, X);
%% show the error
fprintf('data: %s\n', fn_data);
fprintf('model: %s\n', fn_mo);

err = get_cls_err(Ypre, Y);
fprintf('classification error = %d\n', err);
function [X,Y] = load_te_data(fn_data)
load(fn_data);
ind_tr = find( images.set == 3 );

X = images.data(:,:,:, ind_tr);
Y = images.labels(:, ind_tr);

function err = get_cls_err(Ypre, Y)
[~, label_pre] = max(Ypre,[], 1);
[~, label]     = max(Y,[],    1);
N = numel(label);
err = sum( label_pre ~= label ) / N;