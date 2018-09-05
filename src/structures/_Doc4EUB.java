package structures;

/**
 * @author Lin Gong (lg5bt@virginia.edu)
 * Each document has a set of variational parameters:
 *
 */
public class _Doc4EUB extends _Review {

    public _Doc4EUB(int ID, String source, int ylabel, String userID, String productID, String category, long timeStamp) {
        super(ID, source, ylabel, userID, productID, category, timeStamp);
    }

    public void setTopics4Variational(int k, double alpha, double mu, double sigma){

    }
}

