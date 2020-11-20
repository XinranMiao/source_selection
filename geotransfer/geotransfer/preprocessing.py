def read_data(filename):
    bands = loader.S2Bands.ALL
    if isinstance(bands, (list, tuple)):
        bands = [b.value for b in bands]
    else:
        bands = bands.value
    with rasterio.open(filename) as patch:
            data = patch.read(bands)
            bounds = patch.bounds
    return data,bounds

def split_data(metadata,X,y,source_cluster,test_size=0.3,random_state = 1234):
    X_source = X[metadata.clusters == source_cluster]
    X_source = np.array(X_source)
    y_source = y[metadata.clusters == source_cluster]
    y_source = np.array(y_source)
    
    X_target = X[metadata.clusters != source_cluster]
    X_target = np.array(X_target)
    y_target = y[metadata.clusters != source_cluster]
    y_target = np.array(y_target)
    
    X_target_train, X_target_test, y_target_train, y_target_test = train_test_split(X_target ,
                                                                                y_target, 
                                                                                test_size = test_size, 
                                                                                stratify=y_target,
                                                                                random_state = random_state)
    return X_source, y_source, X_target_train, X_target_test,  y_target_train, y_target_test, X_target,y_target


def add_noise(img_arr,mean=0,std=1):
    noisy_img = img_arr + np.random.normal(mean, std, img_arr.shape)
    noisy_img_clipped = np.clip(noisy_img, 0, 255)
    return noisy_img_clipped

def preprocess(img_path):
    img_rz = np.array(Image.open(img_path).resize((256,256)))
    if len(img_rz.shape) == 2:
        img_rz = img_rz.reshape(256,256,1)
        img_13 = np.dstack((add_noise(img_rz), add_noise(img_rz), add_noise(img_rz), add_noise(img_rz),add_noise(img_rz),
                           add_noise(img_rz), add_noise(img_rz), add_noise(img_rz), add_noise(img_rz),add_noise(img_rz),
                           add_noise(img_rz), add_noise(img_rz),add_noise(img_rz)))
    else:
        img_13 = np.dstack((add_noise(img_rz),
                        add_noise(img_rz),
                       add_noise(img_rz),
                       add_noise(img_rz),
                       add_noise(img_rz[:,:,0])))
    img_13 = img_13.astype(np.uint8)
    return img_13