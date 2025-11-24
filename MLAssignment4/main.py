from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy as np
import os


PLOT_DIR = "./plots"
SEED = 42
kvalues = [2,3,4,5]
kmeansInit = ["random", "k-means++"]
covTypes = ["full", "diag"]


def runKmeans(xvalues):
    results = []

    for k in kvalues:
        for init in kmeansInit:
            km = KMeans(n_clusters=k, init=init)
            labels = km.fit_predict(xvalues)
            silScore = silhouette_score(xvalues, labels)
            results.append({
                "k": k,
                "init": init,
                "labels": labels,
                "centers": km.cluster_centers_,
                "silScore": silScore
            })
    return results


def plot_clusters(X, labels, centers=None, title="", save_name=None, silScore="None", centerCount=0, init="init"):
    plt.figure(figsize=(5, 4))
    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=labels,
    )
    if centers is not None:
        plt.scatter(
            centers[:, 0],
            centers[:, 1],
            marker="o",
            linewidth=1,
            edgecolor="k",
            facecolor="none",
        )

    plt.title(title + "SilScore= " + str(silScore) + " k= "+ str(centerCount) + "init= " + init)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.tight_layout()

    if save_name is not None:
        path = os.path.join(PLOT_DIR, save_name)
        plt.savefig(path, dpi=150)
        plt.close()
    else:
        plt.show()

def draw_ellipse(position, covariance, ax=None, n_std=2.0, **kwargs):
    ax = ax or plt.gca()

    cov = covariance

    # Eigenvalues & eigenvectors
    vals, vecs = np.linalg.eigh(cov)
    # Sort eigenvalues (largest first)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    # Width and height of the ellipse (2 * n_std * sqrt(eigenvalues))
    width, height = 2 * n_std * np.sqrt(vals)

    # Angle in degrees of the first eigenvector
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))

    ellipse = Ellipse(
        xy=position,
        width=width,
        height=height,
        angle=angle,
        fill=False
    )
    ax.add_patch(ellipse)

def plot_gmm_clusters(X, labels, model, title="", save_name=None, silScore="None", aLogLikeley=0, bic=0, init="init", k=0):
    plt.figure(figsize=(5, 4))
    ax = plt.gca()

    # Scatter points
    scatter = ax.scatter(
        X[:, 0],
        X[:, 1],
        c=labels,
        s=20,
        cmap="viridis",
        edgecolor="k",
    )

    # Draw an ellipse for each component
    means = model.means_
    covariances = model.covariances_
    cov_type = model.covariance_type

    for i, mean in enumerate(means):
        if cov_type == "full":
            cov = covariances[i]
        elif cov_type == "diag":
            cov = np.diag(covariances[i])

        draw_ellipse(mean, cov, ax=ax, n_std=2.0, linewidth=2)

    ax.set_title("k= " + str(k) +" silScore= "+ str(round(silScore, 2)) + " ALL= " + str(round(aLogLikeley, 2)) + " bic= " + str(round(bic, 2)) + " covar= " + str(init))
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    plt.tight_layout()

    if save_name is not None:
        path = os.path.join(PLOT_DIR, save_name)
        plt.savefig(path, dpi=150)
        plt.close()
    else:
        plt.show()

def run_gmm(X, k_values=kvalues, cov_types=covTypes, random_state=SEED):
    results = []

    for k in k_values:
        for cov_type in cov_types:
            gm = GaussianMixture(
                n_components=k,
                covariance_type=cov_type,
                random_state=random_state,
            )
            gm.fit(X)
            labels = gm.predict(X)

            sil = silhouette_score(X, labels)
            avg_log_like = gm.score(X)    # average log-likelihood per sample
            bic = gm.bic(X)

            results.append({
                "k": k,
                "cov_type": cov_type,
                "labels": labels,
                "silScore": sil,
                "avg_log_like": avg_log_like,
                "bic": bic,
                "model": gm,
            })

    return results


xb, yb = make_blobs(n_samples=100, centers=2, random_state=SEED)

xm, ym = make_moons(n_samples=100,random_state=SEED, noise=0.1)

# this section is responsible for generating the data for the first half of the project.
# blobResults = runKmeans(xb)
# moonResults = runKmeans(xm)

# for result in blobResults:
#     plot_clusters(xb,result["labels"],result["centers"],silScore=result["silScore"], centerCount=result["k"], init=result["init"])

# for result in moonResults:
#     plot_clusters(xm,result["labels"],result["centers"],silScore=result["silScore"], centerCount=result["k"], init=result["init"] )

#this section is responsible for creating the data from the second half of the assignement
blobResults = run_gmm(xb)
moonResults = run_gmm(xm)

# for result in blobResults:
#     plot_gmm_clusters(xb,result["labels"],result["model"],silScore=result["silScore"], aLogLikeley=result["avg_log_like"], bic=result["bic"], init=result["cov_type"], k=result["k"])

for result in moonResults:
    plot_gmm_clusters(xm,result["labels"],result["model"],silScore=result["silScore"], aLogLikeley=result["avg_log_like"], bic=result["bic"], init=result["cov_type"], k=result["k"])