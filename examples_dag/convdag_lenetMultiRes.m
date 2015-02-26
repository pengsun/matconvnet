classdef convdag_lenetMultiRes < convdag
  % Modified LeNet: multi resolution connection at last hiddent layer
  %   The connection is like Fig 2 of "Deep Learning Face Representation from 
  %   Predicting 10,000 Classes" or Fig 1 of "Traffic Signs and Pedestrians
  %   Vision with Multi-Scale Convolutional Networks".
  %   
  %   Warning: No evidence shows such a connection can boost the 
  %   performance on mnist dataset. This wrapper class is just an example
  %   of how to make a non-trivial Directed Acyclic connection.
  
  properties
  end
  
  % override the required functions 
  methods
    function ob = init_dag (ob)
      %%% set the sturcture
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
      % 5: dropout
      tfs{5}   = tf_dropout();
      tfs{5}.i = tfs{4}.o;
      
      % Begin: triangular connection for tfs{5,6,7} 
      % 6: multiplexer
      tfs{6}   = tf_mtx(2);
      tfs{6}.i = tfs{5}.o;
      
      % 7: conv, param
      tfs{7}        = tf_conv();
      tfs{7}.i      = tfs{6}.o(1);
      tfs{7}.p(1).a = f*randn(3,3,50,60, 'single');
      tfs{7}.p(2).a = zeros(1,60,'single');
      
      % 8: concatenator
      tfs{8}      = tf_cat(2);
      tfs{8}.i(1) = tfs{7}.o;
      tfs{8}.i(2) = tfs{6}.o(2);
      % End: triangular connection for tfs{6,7,8} 
      
      % 8: full connection, param
      tfs{9}        = tf_conv();
      tfs{9}.i      = tfs{8}.o;
      tfs{9}.p(1).a = f*randn(1,1,1040,10, 'single');
      tfs{9}.p(2).a = zeros(1,10,'single');
      
      % 10: dropout
      tfs{10}   = tf_dropout();
      tfs{10}.i = tfs{9}.o;
      
      % 11: loss
      tfs{11}      = tf_loss_lse();
      tfs{11}.i(1) = tfs{10}.o;
      
      ob.tfs = tfs;
      
      %%% init the parameters, the optimizers, etc
      ob = prepare_train(ob);
    end
    
    function ob = set_node_src (ob, X_bat, Y_bat)
       ob.tfs{1}.i.a     = X_bat; %
       ob.tfs{11}.i(2).a = Y_bat; %
    end
    
    function ob = set_node_sink (ob, varargin)
      ob.tfs{11}.o.d  = single(1);
    end
    
  end
  
end

