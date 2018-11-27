import grpc

import prediction_pb2
import prediction_pb2_grpc

import numpy as np
import cv2
import datetime
import argparse

#dimentions
w = 224
h = 224
#input = np.zeros((1,10,10),np.float32)
#input = np.reshape(input,(100))


def image_2_vector(input_file):
    nparr = np.fromfile(input_file, dtype=np.float32)
    print("nparr",nparr.dtype,nparr.shape)
    img = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)
    print("img",img.dtype,img.shape)
    print("Initial size",img.shape)
    image = cv2.resize(img, (w, h))
    print("image",image.dtype)
    print("Converted size",image.shape)

    vector = image.reshape((w * h * 3))
    print("vector shape",vector.shape, "vector type", vector.dtype )
    return vector

def image_2_bytes(input_file):
    with open(input_file, "rb") as binary_file:
        # Read the whole file at once
        data = binary_file.read()

        #data = data.tobytes()
        #print(data)
        print("binary data size:", len(data), type(data))
    return data


def run(function,image_path):
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    with grpc.insecure_channel('localhost:5000') as channel:
        stub = prediction_pb2_grpc.ModelStub(channel)
        print("seldon stub", stub)
        start_time = datetime.datetime.now()
        iterations = args['iterations']
        processing_times = np.zeros((0),int)

        if function == "tensor":
            img = cv2.imread(image_path)
            print("img type", type(img))
            print("img",img.shape)
            print("Initial size",img.shape)
            image = cv2.resize(img, (w, h))
            image = image.reshape(1, w, h, 3)
            print("image",image.dtype)
            print("Converted size",image.shape)
            datadef = prediction_pb2.DefaultData(
                names = 'x',
                tensor = prediction_pb2.Tensor(
                    shape = image.shape,
                    values = image.ravel().tolist()
                )
            )
            GRPC_request = prediction_pb2.SeldonMessage(
                data = datadef
            )
            for I in range(iterations):
                start_time = datetime.datetime.now()
                response = stub.Predict(request=GRPC_request)
                end_time = datetime.datetime.now()
                duration = (end_time - start_time).total_seconds() * 1000
                processing_times = np.append(processing_times,np.array([int(duration)]))
        print('processing time for all iterations')
        for x in processing_times:
            print(x,"ms")
        print('processing_statistics')
        print('average time:',round(np.average(processing_times),1), 'ms; average speed:', round(1000/np.average(processing_times),1),'fps')
        print('median time:',round(np.median(processing_times),1), 'ms; median speed:',round(1000/np.median(processing_times),1),'fps')
        print('max time:',round(np.max(processing_times),1), 'ms; max speed:',round(1000/np.max(processing_times),1),'fps')
        print('min time:',round(np.min(processing_times),1),'ms; min speed:',round(1000/np.min(processing_times),1),'fps')
        print('time percentile 90:',round(np.percentile(processing_times,90),1),'ms; speed percentile 90:',round(1000/np.percentile(processing_times,90),1),'fps')
        print('time percentile 50:',round(np.percentile(processing_times,50),1),'ms; speed percentile 50:',round(1000/np.percentile(processing_times,50),1),'fps')
        print('time standard deviation:',round(np.std(processing_times)))
        print('time variance:',round(np.var(processing_times)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Submit gRPC request for inference')
    parser.add_argument('--function', required=True, help='allowed values: tensor')
    parser.add_argument('--image_path',required=True, help='Path to image for classification')
    parser.add_argument('--iterations',required=False, type=int, default=1, help='Number of repetitions for the inference request')
    args = vars(parser.parse_args())

    run(args['function'],args['image_path'])
