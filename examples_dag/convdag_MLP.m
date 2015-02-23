classdef convdag_MLP < convdag
  %CONVDAG_MLP LeNet plus Dropout 
  %   Detailed explanation goes here
  
  properties
  end
  
  % override the required functions 
  methods
    function ob = init_dag (ob)
      
      %%% 784 -- 1000 -- 800 -- 10
      
      f = 1/100;
      %%% Layer I
      % 1: full connection, param
      h = tf_conv();
      h.i = n_data();
      h.o = n_data();
      h.p = [n_data(), n_data()];
      h.p(1).a = f*randn(28,28,1,1000, 'single'); % kernel
      h.p(2).a = zeros(1, 1000, 'single'); % bias
      tfs{1} = h;
      % 2: relu
      h = tf_relu();
      h.i = tfs{1}.o;
      h.o = n_data();
      tfs{2} = h;
      % 3: dropout 
      h = tf_dropout();
      h.i = tfs{2}.o;
      h.o = n_data();
      tfs{3} = h;
      
      %%% Layer II
      % 4: full connection, param
      h = tf_conv();
      h.i = tfs{3}.o;
      h.o = n_data();
      h.p = [n_data(), n_data()];
      h.p(1).a = f*randn(1,1,1000,800, 'single'); % kernel
      h.p(2).a = zeros(1, 800, 'single'); % bias
      tfs{4} = h;
      % 5: relu
      h = tf_relu();
      h.i = tfs{4}.o;
      h.o = n_data();
      tfs{5} = h;
      % 6: dropout 
      h = tf_dropout();
      h.i = tfs{5}.o;
      h.o = n_data();
      tfs{6} = h;
      
      %%% Layer III
      % 7: full connection, param
      h = tf_conv();
      h.i = tfs{6}.o;
      h.o = n_data();
      h.p = [n_data(), n_data()];
      h.p(1).a = f*randn(1,1,800,10, 'single'); % kernel
      h.p(2).a = zeros(1, 10, 'single'); % bias
      tfs{7} = h;
      % 8: relu
      h = tf_relu();
      h.i = tfs{7}.o;
      h.o = n_data();
      tfs{8} = h;

      %%% the loss
      % 9: loss
      h = tf_loss_lse();
      h.i = [tfs{8}.o, n_data()];
      h.o = n_data();
      tfs{9} = h;
      
      ob.tfs = tfs;
    end
    
    function ob = set_node_src (ob, X_bat, Y_bat)
       ob.tfs{1}.i.a    = X_bat; %
       ob.tfs{9}.i(2).a = Y_bat; %
    end
    
    function ob = set_node_sink (ob, varargin)
      ob.tfs{9}.o.d  = single(1);
    end
    
  end
  
end

