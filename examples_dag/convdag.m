classdef convdag
  %convdag A thin wrapper for convolutional DAG
  %   Detailed explanation goes here
  
  % options
  properties
    beg_epoch; % beggining epoch
    num_epoch; % number of epoches
    batch_sz; % batch size
    dir_mo; % directory for models
    L_tr; % training loss
    
    % TODO: more properties: step size, momentum...
  end
  
  properties
    tfs; % transformer array
    opt_arr; % numeric optimization array, one for each params{i}
    params;  % parameters array, linked to those in tfs via handle class
  end
  
  methods
    function ob = convdag()
      ob.beg_epoch = 1; % begining epoch
      ob.num_epoch = 5; % number of epoches
      ob.batch_sz = 128; % batch size
      ob.dir_mo = './mo_zoo/foobar'; % directory for models
      
    end
    
    function ob = train (ob, X, Y)
    % train with instance X and lables Y
    % Input:
    %   X: [d1,d2,d3, N], where N = #instances, d1,...,dm are dims
    %   Y: [K, N], where K is #dims of the labels
    %
      
      %%% initialize the dag before calling train() 
      %%% do this by calling init_dag() or load the model from file
      
      ob = prepare_train (ob);
      
      for t = ob.beg_epoch : ob.num_epoch
        % fire: train one epoch
        ob = prepare_train_one_epoch(ob, t);
        ob = train_one_epoch(ob, X,Y);
        ob = post_train_one_epoch(ob, t, size(X,4));
        
        % save the result
        fn_cur_mo = fullfile(ob.dir_mo, sprintf('dag_epoch_%d.mat',t) );
        ob = save_model (ob, fn_cur_mo);
      end % for t
      
    end % train
    
    function Ypre = test (ob, X)
      
      % prepare
      ob = prepare_test(ob);
      
      % initialize a batch generator
      hbat = bat_gentor();
      N = size(X, 4);
      hbat = reset(hbat, N, ob.batch_sz);
      
      % test every batch
      % What? Why divide the testing set into batches? Becuuse this would
      % generate many printings that relieve you while you watch the screen
      for i_bat = 1 : hbat.num_bat
        t_elapsed = tic; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     
        % get batch 
        ind = get_idx_orig(hbat, i_bat);
        X_bat = X(:,:,:, ind);
        Y_bat_trash = 0; % Just making the fprop() goes. Okay with a scalar.
        
        % set source nodes
        ob = set_node_src(ob, X_bat, Y_bat_trash);
        
        % fire: do the batch testing by calling fprop() on each transformer
        ob.tfs = cellfun(@fprop, ob.tfs, 'uniformoutput',false);
        
        % fetch the results
        Ypre_bat = squeeze( ob.tfs{end-1}.o.a );
        if (i_bat==1), Ypre = Ypre_bat;
        else           Ypre = cat(2,Ypre,Ypre_bat); end
        t_elapsed = toc(t_elapsed); %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % print 
        fprintf('testing: epoch %d, batch %d of %d, ',...
          ob.opt_arr(1).cc.epoch_cnt, i_bat, hbat.num_bat);
        fprintf('time = %.3fs, speed = %.0f images/s\n',...
          t_elapsed, ob.batch_sz/t_elapsed);
        
      end % for ii
      
    end % test
  end % methods
  
  methods % auxiliary functions for train
    function ob = prepare_train (ob)
      
      % set context
      for i = 1 : numel( ob.tfs )
        ob.tfs{i}.cc.is_tr = true;
      end
      % the parameters and the corresponding numeric optimizers
      if ( numel(ob.params) ~= numel(ob.tfs) ) % otherwise the
        ob.params = dag_util.collect_params(ob.tfs);
        ob.opt_arr = dag_util.alloc_opt( numel(ob.params) );
      end
      %
      if ( ~exist(ob.dir_mo, 'file') ), mkdir(ob.dir_mo); end
    end % prepare_train
    
    function ob = prepare_train_one_epoch (ob, i_epoch)
      % set calling context
      for i = 1 : numel(ob.opt_arr)
        ob.opt_arr(i).cc.epoch_cnt = i_epoch;
      end % for i
      
      % update the loss
      ob.L_tr(i_epoch) = 0;
    end % prepare_train_one_epoch
    
    function ob = train_one_epoch (ob, X,Y)
    % train one epoch
    
      % initialize a batch index generator
      hbat = bat_gentor();
      N = size(X, 4);
      hbat = reset(hbat, N, ob.batch_sz);
      
      % train every batch
      for i_bat = 1 : hbat.num_bat
        t_elapsed = tic; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     
        % get batch 
        ind = get_idx(hbat, i_bat);
        X_bat = X(:,:,:, ind);
        Y_bat = Y(:, ind);
        
        % set source nodes, sink nodes
        ob = set_node_src(ob, X_bat, Y_bat);
        ob = set_node_sink(ob);
        
        % fire: do the batch training
        ob = prepare_train_one_bat(ob, i_bat);
        ob = train_one_bat(ob);
        ob = post_train_one_bat(ob, i_bat);
        t_elapsed = toc(t_elapsed); %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % print 
        fprintf('epoch %d, batch %d of %d, ',...
          ob.opt_arr(1).cc.epoch_cnt, i_bat, hbat.num_bat);
        fprintf('time = %.3fs, speed = %.0f images/s\n',...
          t_elapsed, ob.batch_sz/t_elapsed);
        
      end % for ii
    
    end % train_one_eporch
    
    function ob = post_train_one_epoch (ob, i_epoch, varargin)
      % normalize the loss
      N = varargin{1};
      ob.L_tr(end) = ob.L_tr(end) ./ N; 
    end % post_train_one_epoch
    
    function ob = prepare_train_one_bat (ob, i_bat)
      % set calling context
      for i = 1 : numel(ob.opt_arr)
        ob.opt_arr(i).cc.batch_sz = ob.batch_sz;
        ob.opt_arr(i).cc.iter_cnt = i_bat;
      end % for i
    end % prepare_train_one_bat
    
    function ob = train_one_bat (ob)
    % train one batch
    
      % fprop & bprop
      ob.tfs           = cellfun(@fprop, ob.tfs,...
        'uniformoutput',false);
      ob.tfs(end:-1:1) = cellfun(@bprop, ob.tfs(end:-1:1),...
        'uniformoutput',false);
      
      % update parameters
      for i = 1 : numel(ob.opt_arr)
        ob.opt_arr(i) = update(ob.opt_arr(i), ob.params(i));
      end
    end % train_one_bat
    
    function ob = post_train_one_bat (ob, i_bat)
      % update the loss
      LL = ob.tfs{end}.o.a;
      ob.L_tr(end) = ob.L_tr(end) + sum(LL(:));
    end % post_train_one_bat
    
    function ob = clear_im_data (ob)
    % clear the intermediate (unnecessary) data: hidden variables .a, .d
    % parameters .d
      
      % clear the input for each transformer
      ob.tfs = cellfun(@cl_io, ob.tfs, 'uniformoutput',false);
      
      % clear .d for all parameters
      for k = 1 : numel( ob.params )
        ob.params(k).d = [];
      end
      
    end % clear_im_data
    
    function ob = save_model(ob, fn)
      ob = clear_im_data(ob);
      save(fn, 'ob');
    end % save_model
    
  end % methods
  
  methods % auxiliary functions for test
    function ob = prepare_test(ob)
      for i = 1 : numel(ob.tfs)
        ob.tfs{i}.cc.is_tr = false;
      end % for i
    end % prepare_test
  end % methods
    
  methods % need be overrided in derived class
    function ob = init_dag (ob)
      error('Call this in a derived class :)');
    end
    
    function ob = set_node_src (ob, X_bat, Y_bat)
      error('Call this in a derived class :)');
    end
    
    function ob = set_node_sink (ob, varargin)
      error('Call this in a derived class :)');
    end

  end % methods
  
end % convdag

