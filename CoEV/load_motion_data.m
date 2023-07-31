function [ data ] = load_motion_data(affine)

datadir = './Hopkins155';
seqs = dir(datadir);
seq3 = seqs(3 : end);
data = struct('X',{},'name',{},'ids',{});
for i=1:length(seq3)
    fname = seq3(i).name;
    fdir = [datadir '/' fname];
    if isdir(fdir)
        datai = load([fdir '/' fname '_truth.mat']);
        id = length(data)+1;
        data(id).ids = datai.s;
        data(id).name = lower(fname);
        X = reshape(permute(datai.x(1:2,:,:),[1 3 2]),2 * datai.frames, datai.points);
        if affine == 1
            data(id).X = [X; ones(1,size(X,2))];
        else
            data(id).X = X;
        end
    end
end
clear seq3;

end

