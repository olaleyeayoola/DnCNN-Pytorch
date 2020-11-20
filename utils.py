from torch.autograd import Variable


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg",
                                                              ".jpeg", ".gif"])


def gaussian(ins, is_training, mean, stddev):
    if is_training:
        noise = Variable(ins.data.new(ins.size()).normal_(mean, stddev))
        return ins + noise
    return ins
