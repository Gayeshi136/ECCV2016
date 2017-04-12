%% function to do power normalization

function normfeature = func_L2Normalization(feature)

if iscell(feature)
    numSample = numel(feature);
    numDim = numel(feature{1});
    normfeature = zeros(numDim,numSample);
    
    parfor i=1:numel(feature)
        
        %     signs = sign(feature{i});
        
        try normfeature(i,:) = feature{i}./norm(feature{i});
        catch
            normfeature(i,:) = zeros(numDim,1);
        end
        
    end
    
    normfeature = normfeature';
    
else
    numSample = size(feature,1);
    numDim = size(feature,2);
    normfeature = zeros(numSample,numDim);
    
    normfeature = feature./repmat(sqrt(sum(feature.^2,2))+eps,1,size(feature,2));
    
%     for i = 1:size(feature,1)
%         try normfeature(i,:) = feature(i,:)./norm(feature(i,:));
%         catch
%             normfeature(i,:) = zeros(numDim,1);
%         end
%     end
end