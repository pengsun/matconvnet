classdef dag_util
%DAG_UTIL Encapsulate Utility Functions
%   Detailed explanation goes here
methods(Static)
  
  function params = collect_params(tfs)
    params = [];
    for i = 1 : numel(tfs)
      % empty
      if ( isempty(tfs{i}.p) ), continue; end
      
      % tf: fecth them
      if ( isempty(params) )
        params = tfs{i}.p;
      else
        params = [params, tfs{i}.p];
      end
    end % for i
  end % collect_params
  
  function opt_arr = alloc_opt(Nparams)

    for i = 1 : Nparams
      opt_arr(i)    = opt_1storder();
      opt_arr(i).cc = dag_util.create_cc();
    end % for i
  end % alloc_opt
  
  function cc = create_cc()
  % create calling context  
    cc.iter_cnt = 1;
    cc.epoch_cnt = 1;
    cc.batch_sz = 1;
    cc.is_tr = true;
  end
  
  
end % methods(Static)
end % classdef dag_util

