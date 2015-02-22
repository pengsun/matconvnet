classdef dag_util
%DAG_UTIL Summary of this class goes here
%   Detailed explanation goes here
methods(Static)
  
  function params = collect_params(tfs)
    params = {};
    for i = 1 : numel(tfs)
      if ( isempty(tfs{i}.p) ), continue; end
      for j = 1 : numel( tfs{i}.p )
        params{end+1} = tfs{i}.p(j); %#ok<AGROW>
      end % for j
    end % for i
  end % collect_params
  
  
  
end % methods(Static)
end % classdef dag_util

