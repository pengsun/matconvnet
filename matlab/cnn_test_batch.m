function [y, res] = cnn_test_batch(net, x)
% CNN_TEST_BATCH test just one instance or instance batch (instances)

% % setup toolbox
% run(fullfile(fileparts(mfilename('fullpath')), ...
%   '..', 'matlab', 'vl_setupnn.m')) ;

% run the CNN for testing
res = vl_simplenn(net, x, [], [], ...
  'disableDropout', true ) ;

% decide the prediction layer
ell_prev = 0;
tp = net.layers{end}.type;
if ( strcmp(tp,'lse') || ...
     strcmp(tp,'softmaxloss') || ...
     strcmp(tp,'loss') )
   ell_prev = 1;
end
% return the prediction
y = squeeze(gather(res(end-ell_prev).x)) ;

