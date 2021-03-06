classdef convdag
  %convdag A thin wrapper for convolutional DAG
  %   Detailed explanation goes here
  
  % options
  properties
    beg_epoch; % beggining epoch
    num_epoch; % number of epoches
    batch_sz; % batch size
    dir_mo; % directory for models
    is_tightMem; % if tight memory?
  end
  
  properties
    tfs; % transformer array
    opt_arr; % numeric optimization array, one for each params(i)
    params;  % parameters array, linked to those in tfs via handle class
    
    L_tr; % training loss
    cc; % calling context
  end
  
  methods
    function ob = convdag()
      ob.beg_epoch = 1; % begining epoch
      ob.num_epoch = 5; % number of epoches
      ob.batch_sz = 128; % batch size
      ob.dir_mo = './mo_zoo/foobar'; % directory for models
      ob.is_tightMem = false;
      
      ob.cc = call_cntxt();
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
          ob.cc.epoch_cnt, i_bat, hbat.num_bat);
        fprintf('time = %.3fs, speed = %.0f images/s\n',...
          t_elapsed, ob.batch_sz/t_elapsed);
        
      end % for ii
      
    end % test
  end % methods
  
  methods % auxiliary functions for train
    function ob = prepare_train (ob)

      % the parameters and the corresponding numeric optimizers
      if ( numel(ob.params) ~= numel(ob.tfs) ) % otherwise the
        ob.params = dag_util.collect_params(ob.tfs);
        %ob.opt_arr = dag_util.alloc_opt( numel(ob.params) );
        ob.opt_arr = opt_1storder();
        ob.opt_arr( numel(ob.params) ) = opt_1storder();
      end
      
      %%% set calling context
      for i = 1 : numel( ob.tfs )
        ob.tfs{i}.cc = ob.cc;
      end
      for k = 1 : numel(ob.opt_arr)
        ob.opt_arr(k).cc = ob.cc;
      end
      % training
      ob.cc.is_tr = true;
      
      %
      if ( ~exist(ob.dir_mo, 'file') ), mkdir(ob.dir_mo); end
    end % prepare_train
    
    %%% for training one epoch
    function ob = prepare_train_one_epoch (ob, i_epoch)
      % set calling context
      ob.cc.epoch_cnt = i_epoch;

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
          ob.cc.epoch_cnt, i_bat, hbat.num_bat);
        fprintf('time = %.3fs, speed = %.0f images/s\n',...
          t_elapsed, ob.batch_sz/t_elapsed);
        
      end % for ii
    
    end % train_one_eporch
    
    function ob = post_train_one_epoch (ob, i_epoch, varargin)
      % normalize the loss
      N = varargin{1};
      ob.L_tr(end) = ob.L_tr(end) ./ N; 
    end % post_train_one_epoch
    
    %%% for traing one batch
    function ob = prepare_train_one_bat (ob, i_bat)
      % set calling context
      ob.cc.batch_sz = ob.batch_sz;
      ob.cc.iter_cnt = i_bat;
    end % prepare_train_one_bat
    
    function ob = train_one_bat (ob)
    % train one batch
    
      % fprop & bprop
      if (ob.is_tightMem)
        % TODO: modify it when vl_feat/matconvnet fully supports it
        warning('Tight memory has not been fully supported by original matconvnet!');
        ob = do_fprop_bprop(ob);
        %ob = do_fprop_bprop_tight_mem(ob);
      else
        ob = do_fprop_bprop(ob);
      end
      
      % update parameters
      for i = 1 : numel(ob.opt_arr)
        ob.opt_arr(i) = update(ob.opt_arr(i), ob.params(i));
      end
    end % train_one_bat
    
    function ob = do_fprop_bprop(ob)
      ob.tfs           = cellfun(@fprop, ob.tfs,...
        'uniformoutput',false);
      ob.tfs(end:-1:1) = cellfun(@bprop, ob.tfs(end:-1:1),...
        'uniformoutput',false);
    end % do_fprop_bprop
    
    function ob = do_fprop_bprop_tight_mem(ob)
      % fprop
      for i = 1 : numel(ob.tfs)
        ob.tfs{i} = fprop( ob.tfs{i} );
        ob.tfs{i} = cl_i_a( ob.tfs{i} );
      end % for i
      
      % bprop
      for i = numel(ob.tfs) : -1 : 1
        ob.tfs{i} = bprop( ob.tfs{i} );
        ob.tfs{i} = cl_o_d( ob.tfs{i} );
      end
    end % do_fprop_bprop_tight_mem
    
    function ob = post_train_one_bat (ob, i_bat)
      % update the loss
      LL = gather( ob.tfs{end}.o.a ); % cpu or gpu array
      ob.L_tr(end) = ob.L_tr(end) + sum(LL(:));
    end % post_train_one_bat
    
    %%% for internal data management
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
      ob.cc.is_tr = false;
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

