function ap = func_CalcMap(confs, labelsVid, IX)
%confs = confs + 0.000001*rand(size(confs));
if isrow(confs)
    confs = confs';
end

if ~exist('IX','var')
    [confS IX] = sort(confs, 'descend');
end
labels = labelsVid(IX);

numOfRels = 0;
ap = 0;
for i = 1 : numel(labels)
   if labels(i) == 1
       numOfRels = numOfRels + 1;
       ap = ap + (numOfRels ./ i);
   end
end
ap = ap ./ numOfRels;

end

