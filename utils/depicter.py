def sic(x,y):
    ckmeans_initial = SeededKMeans(n_clusters=2).fit(x, y)
    labels = ckmeans_initial.labels_
    score_prev = 0
    while True:
        try:
            ckmeans_iter = SeededKMeans(n_clusters=2).fit(x[labels==1], y[labels==1])
            labels_temp = np.zeros(len(x))
            labels_temp[labels==1] = ckmeans_iter.labels_
        except FloatingPointError:
            if (y[labels==1]==1).any() and x[labels==1].shape[0]>1:
                    kmeans = KMeans(n_clusters=2).fit(x[labels==1])
                    if np.mean(kmeans.labels_^1==y[labels==1])> np.mean(kmeans.labels_==y[labels==1]):
                        kmeans_labels = kmeans.labels_^1
                    else: 
                        kmeans_labels = kmeans.labels_
                    labels_temp = np.zeros(len(x))
                    labels_temp[labels==1] = kmeans_labels
            break
        except EmptyClustersException:
            break
        score = accuracy_score(y[y>-1], labels_temp[y>-1])
        if score<=score_prev:
            break
        else:
            score_prev = score
            labels = labels_temp
    return labels
