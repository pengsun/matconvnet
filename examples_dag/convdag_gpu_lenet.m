classdef convdag_gpu_lenet < convdag_gpu
  %CONVDAG_LENET LeNet 
  %   Detailed explanation goes here
  
  properties
  end
  
  % override the required functions 
  methods
    function ob = init_dag (ob)
      %%% set the sturcture
      f = 1/100;
      % 1: conv, param
      tfs{1}        = tf_conv();
      tfs{1}.p(1).a = gpuArray( f*randn(5,5,1,20, 'single') ); % kernel
      tfs{1}.p(2).a = gpuArray( zeros(1, 20, 'single') ); % bias
      % 2: pool
      tfs{2}   = tf_pool();
      tfs{2}.i = tfs{1}.o;
      % 3: conv, param
      tfs{3}        = tf_conv();
      tfs{3}.i      = tfs{2}.o;
      tfs{3}.p(1).a = gpuArray( f*randn(5,5,20,50, 'single') );
      tfs{3}.p(2).a = gpuArray( zeros(1,50,'single') );
      % 4: pool
      tfs{4}   = tf_pool();
      tfs{4}.i = tfs{3}.o;
      % 5: full connection, param
      tfs{5}        = tf_conv();
      tfs{5}.i      = tfs{4}.o;
      tfs{5}.p(1).a = gpuArray( f*randn(4,4,50,500, 'single') );
      tfs{5}.p(2).a = gpuArray( zeros(1,500,'single') );
      % 6: relu
      tfs{6}   = tf_relu();
      tfs{6}.i = tfs{5}.o;
      % 7: full connection, param
      tfs{7}        = tf_conv();
      tfs{7}.i      = tfs{6}.o;
      tfs{7}.p(1).a = gpuArray( f*randn(1,1,500,10, 'single') );
      tfs{7}.p(2).a = gpuArray( zeros(1,10,'single') );
      % 8: loss
      tfs{8}      = tf_loss_lse();
      tfs{8}.i(1) = tfs{7}.o;
      
      ob.tfs = tfs;
      
      %%% init the parameters, the optimizers, etc
      ob = prepare_train(ob);
    end
    
    function ob = set_node_src (ob, X_bat, Y_bat)
       ob.tfs{1}.i.a    = X_bat; %
       ob.tfs{8}.i(2).a = Y_bat; %
       
       ob.tfs{1}.i.a    = gpuArray( ob.tfs{1}.i.a );
       ob.tfs{8}.i(2).a = gpuArray( ob.tfs{8}.i(2).a ); %
    end
    
    function ob = set_node_sink (ob, varargin)
      ob.tfs{8}.o.d  = gpuArray( single(1) );
    end
    
  end
  
end

