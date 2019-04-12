package topicmodels.RoleEmbedding;

public class UserEmbeddingSkipGram extends UserEmbeddingBaseline {
    public UserEmbeddingSkipGram(int m, int nuIter, double converge, double alpha, double stepSize){
        super(m, nuIter, converge, alpha, stepSize);
    }
}
