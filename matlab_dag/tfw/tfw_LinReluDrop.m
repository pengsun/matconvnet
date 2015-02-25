classdef tfw_LinReluDrop < tfw_i
  %TFW_LINRELUDROP Linear layer + Relu + Dropout
  %   Detailed explanation goes here
  
  methods
    function ob = tfw_LinReluDrop(sz)
    % Input:
    %  sz: [W, H, C, M]. 
    
      %%% internal connection
      % 1: full connection, param
      h = tf_conv();
      h.i = n_data();
      h.o = n_data();
      h.p = [n_data(), n_data()];
      h.p(1).a = f*randn(sz, 'single'); % kernel
      h.p(2).a = zeros(1, sz(end), 'single'); % bias
      ob.tfs{1} = h;
      
      % 2: relu
      h = tf_relu();
      h.i = ob.tfs{1}.o;
      h.o = n_data();
      ob.tfs{2} = h;
      
      % 3: dropout
      h = tf_dropout();
      h.i = ob.tfs{2}.o;
      h.o = n_data();
      ob.tfs{3} = h;
      
      %%% link the input/output of tfw to that of internal connection
      ob.i = ob.tfs{1}.i;
      ob.o = ob.tfs{3}.o;
      
      %%% set the parameters
      ob.p = dag_util.collect_params( ob.tfs );
    end % tfw_LinReluDrop
  end
  
end

