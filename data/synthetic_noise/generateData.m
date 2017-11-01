function [ data_majority, data_minority, data_noise ] = generateData( majoritySize, minoritySize, noiseRate )
%Synthesize data in NIPs 2013 paper "Learning with Noisy Labels"
%   Input:  majoritySize -- number of data in the majority class;
%           minoritySize -- number of data in the minority class;
%           noiseRate - percentage of majority data with flipped class.
%   Output: data_majority -- n*2 matrix contains n majority data,
%                            n = majoritySize-majoritySize*noiseRate;
%           data_minority -- m*2 matrix contains m minority data,
%                            m = minoritySize;
%           data_noise -- k*2 matrix contains k noise data originated from
%                         majority, k= majoirtySize*noiseRate.
%
%   The three data matrices are saved as .dat files.
%
%Example:
%   [ data_majority, data_minority, data_noise ] = generateDate( 2000, 100, 0.1 );
%   will generate 100 minority data, 2000*(1-0.1)=1800 majority data, and 2000*0.1=200 noise data.
%   A plot will also be displayed showing majority in blue circle, minority in red circle, and noise in red cross.


data_majority=(rand(majoritySize,2)-0.5)*200;
index_xbigger=find(data_majority(:,1)>data_majority(:,2));
data_majority(index_xbigger,[1,2])=data_majority(index_xbigger,[2,1]);

data_minority=rand(minoritySize,2);
index_ybigger=find(data_minority(:,2)>data_minority(:,1));
data_minority(index_ybigger,[1,2])=data_minority(index_ybigger,[2,1]);
data_minority(:,1)=(data_minority(:,1)-0.2)*125;
data_minority(:,2)=(data_minority(:,2)-0.8)*125;

noiseSize=round(majoritySize*noiseRate);
index_flip=randi(majoritySize,noiseSize,1);
data_noise=data_majority(index_flip,:);
data_majority(index_flip,:)=[];

csvwrite('data_majority.csv',data_majority);
csvwrite('data_minority.csv',data_minority);
csvwrite('data_noise.csv',data_noise);


figure;
scatter(data_majority(:,1),data_majority(:,2),'o','b');
hold on;
scatter(data_minority(:,1),data_minority(:,2),'o','r');
scatter(data_noise(:,1),data_noise(:,2),'+','r');

end

