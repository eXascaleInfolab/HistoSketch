function [ res ] = randomHash_fast_fixlen( x, seed, len )
% random hash function with fixed range output using Jenkins hash function


temp = jenkinshash(num2str(x),32);    
res = bitxor(temp,seed,'uint32');
res = mod(res, len) + 1;

end

