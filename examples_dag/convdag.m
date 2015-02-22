classdef convdag
  %convdag Summary of this class goes here
  %   Detailed explanation goes here
  
  % options
  properties
    num_epoch; % number of epoches
    batch_sz; % batch size
    dir_mo; % directory for models
  end
  
  properties
    tfs; % transformer array
    opt_arr; % numeric optimization array, one for each params{i}
    params;  % parameters array, linked to those in tfs via handle class
  end
  
  methods
    function ob = convdag()
      ob.num_epoch = 5; % number of epoches
      ob.batch_sz = 128; % batch size
      ob.dir_mo = './mo_zoo/foobar'; % directory for models
      
    end
    
    function ob = train (ob, X, Y)
    % train with instance X and lables Y
    % Input:
    %   X: [d1,d2,...,dm, N], where N = #instances, d1,...,dm are dims
    %   Y: [K, N], where K is #dims of the labels
    %
    
      %%% prepare to training
      % initialize: do it before calling train(:) 
      %   ob = init_connection(ob); 
      % or load the dag from saved file, e.g., 
      %   load('dab_epoch_42.mat');
      %
      % collect the parameters from the transformer array
      ob.params = dag_util.collect_params(ob.tfs);
      % create the corresponding numeric optimizers
      ob.opt_arr = dag_util.alloc_opt(ob.params);
      
      %%% train with SGD
      for t = 1 : ob.num_epoch
        
        % set calling context
        for i = 1 : numel(ob.opt_arr)
          ob.opt_arr{i}.cc.epoch_cnt = t;
        end % for i
        
        % train one epoch
        ob = train_one_eopch(ob, X,Y);
        
        % save the result
        % TODO: delete the unncessary data before saving, only params are
        % ineterested!!!
        % TODO: save info holding training objective, errors, etc.
        save(fn_cur_mo, 'ob');
      end % for t
      
    end % train
    
    function ob = train_one_epoch (ob, X,Y)
    % train one epoch
    
      % initialize a batch index generator
      hbat = bat_gentor();
      N = size(X, 4);
      hbat = reset(hbat, N, ob.batch_sz);
      
      % train every batch
      for ii = 1 : hbat.num_bat
        t_elapsed = tic; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     
        % get batch 
        ind = get_idx(hbat, ii);
        X_bat = X(:,:,:, ind);
        Y_bat = Y(:, ind);
        
        % set source nodes, sink nodes
        ob = set_node_src(ob, X_bat, Y_bat);
        ob = set_node_sink(ob);
        
        % set calling context
        for i = 1 : numel(ob.opt_arr)
          ob.opt_arr{i}.cc.batch_sz = ob.batch_sz;
          ob.opt_arr{i}.cc.iter_cnt = ii;
        end % for i
        
        % fire: do the batch training
        ob = train_one_bat(ob);
        t_elapsed = toc(t_elapsed); %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % print 
        fprintf('batch %d of %d, time = %.3fs, speed = %.1f images/s\n',...
          ii, hbat.num_bat,  t_elapsed, ob.batch_sz/t_elapsed);
        
      end % for ii
    
    end % train_one_eporch
    
    function ob = train_one_bat (ob)
    % train one batch
    
      % fprop & bprop
      ob.tfs           = cellfun(@fprop, ob.tfs,...
        'uniformoutput',false);
      ob.tfs(end:-1:1) = cellfun(@bprop, ob.tfs(end:-1:1),...
        'uniformoutput',false);
      
      % update parameters
      ob.opt_arr = arrayfun(@update, ob.opt_arr, ob.params,...
        'uniformoutput',false);
    end % train_one_bat
    
  end % methods
    
  methods % need be overwritten in derived class
    function ob = init_connection (ob)
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

