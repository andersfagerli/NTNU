from ransac import Ransac
from models import RansacTest
import matplotlib.pyplot as plt

if __name__ == "__main__":
    lin_model = RansacTest(size=300, num_outliers=150)
    sample_size = 2
    threshold = 1
    min_model_votes = 45
    ransac = Ransac(lin_model, sample_size, threshold, min_model_votes)
    best_fit, best_err, improved_sample_idx = ransac.run()
    print(best_fit)
    print(best_err)
    print(improved_sample_idx)
    plt.plot(lin_model.x, lin_model.y, '.', label='True data')
    plt.plot(lin_model.x, best_fit[0]*lin_model.x + best_fit[1], label='RANSAC')
    plt.plot(lin_model.x, lin_model.x, label='True model')
    plt.legend()
    plt.show()
    
