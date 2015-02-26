classdef convdag_MLP2 < convdag
  %CONVDAG_MLP LeNet plus Dropout 
  %   Detailed explanation goes here
  
  properties
  end
  
  % override the required functions 
  methods
    function ob = init_dag (ob)
      
      %%% init the connection: 784 -- 1000 -- 800 -- 10
      f = 1/100;
      % Layer I
      % 1: full connection, param
      sz1 = [28,28,1,1000];
      tfs{1} = tfw_LinReluDrop(sz1); 
      
      % Layer II
      % 2: full connection, param
      sz2 = [1,1,1000,800];
      tfs{2}   = tfw_LinReluDrop(sz2);
      tfs{2}.i = tfs{1}.o;
      
      % Layer III
      % 3: full connection, output, param
      tfs{3}        = tf_conv();
      tfs{3}.i      = tfs{2}.o;
      tfs{3}.p(1).a = f*randn(1,1,800,10, 'single'); % kernel
      tfs{3}.p(2).a = zeros(1, 10, 'single'); % bias
      
      % the loss
      % 4: loss
      tfs{4}      = tf_loss_lse();
      tfs{4}.i(1) = tfs{3}.o;
      
      ob.tfs = tfs;
      
      %%% init the parameters, the optimizers, etc
      ob = prepare_train(ob);
    end
    
    function ob = set_node_src (ob, X_bat, Y_bat)
       ob.tfs{1}.i.a    = X_bat; %
       ob.tfs{4}.i(2).a = Y_bat; %
    end
    
    function ob = set_node_sink (ob, varargin)
      ob.tfs{4}.o.d  = single(1);
    end
    
  end % methods
  
end % convdag_MLP2