package topicmodels.UserEmbedding;

import structures.*;
import topicmodels.LDA.LDA_Variational;

import java.util.ArrayList;
import java.util.Collection;

/**
 * @author Lin Gong (lg5bt@virginia.edu)
 * The joint modeling of user embedding (U*M) and topic embedding (K*M)
 */

public class EUB extends LDA_Variational {

    ArrayList<_Topic4EUB> m_topics;
    ArrayList<_User4EUB> m_users;

    protected int m_embedding_dim;

    /*****variational parameters*****/
    protected double d_mu = 1.0, d_sigma_theta = 1.0;
    protected double u_mu = 1.0, d_sigma_P = 1.0;

    public EUB(int number_of_iteration, double converge, double beta,
               _Corpus c, double lambda, int number_of_topics, double alpha,
               int varMaxIter, double varConverge, int m) {
        super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha,
                varMaxIter, varConverge);
        m_embedding_dim = m;
    }

    @Override
    public void EMonCorpus(){

        m_trainSet = new ArrayList<>();
        // collect all training reviews
        for(_User u: m_users){
            for(_Review r: u.getTrainReviews()){
                m_trainSet.add(r);
            }
        }
        EM();
    }

    @Override
    public void EM(){
        initialize_probability();
    }

    @Override
    protected void initialize_probability(Collection<_Doc> collection) {

        for(_Topic4EUB t: m_topics){
            t.setTopics4Variational(m_embedding_dim, m_mu, m_sigma);
        }
        for(_Doc d: m_trainSet)
            d.setTopics4Variational(m_embedding_dim, d_alpha, );
        for(_User u: m_users)
            u.setTopics4Variational();

        calculate_M_step(0);

    }

    protected void initDoc(_Doc d){

    }





}