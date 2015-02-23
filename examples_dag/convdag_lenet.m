classdef convdag_lenet < convdag
  %CONVDAG_LENET LeNet 
  %   Detailed explanation goes here
  
  properties
  end
  
  % override the required functions 
  methods
    function ob = init_dag (ob)
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
      % 6: relu
      h = tf_relu();
      h.i = tfs{5}.o;
      h.o = n_data();
      tfs{6} = h;
      % 7: full connection, param
      h = tf_conv();
      h.i = tfs{6}.o;
      h.o = n_data();
      h.p = [n_data(), n_data()];
      h.p(1).a = f*randn(1,1,500,10, 'single');
      h.p(2).a = zeros(1,10,'single');
      tfs{7} = h;
      % 8: loss
      h = tf_loss_lse();
      h.i = [tfs{7}.o, n_data()];
      h.o = n_data();
      tfs{8} = h;
      
      ob.tfs = tfs;
    end
    
    function ob = set_node_src (ob, X_bat, Y_bat)
       ob.tfs{1}.i.a    = X_bat; %
       ob.tfs{8}.i(2).a = Y_bat; %
    end
    
    function ob = set_node_sink (ob, varargin)
      ob.tfs{8}.o.d  = single(1);
    end
    
  end
  
end

