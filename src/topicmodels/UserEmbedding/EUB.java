package topicmodels.UserEmbedding;

import structures.*;
import topicmodels.LDA.LDA_Variational;

import java.util.ArrayList;
import java.util.Arrays;
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
    protected double t_mu = 1.0, t_sigma = 1.0;
    protected double d_mu = 1.0, d_sigma = 1.0;
    protected double u_mu = 1.0, u_sigma = 1.0;


    public EUB(int number_of_iteration, double converge, double beta,
               _Corpus c, double lambda, int number_of_topics, double alpha,
               int varMaxIter, double varConverge, int m) {
        super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha,
                varMaxIter, varConverge);
        m_embedding_dim = m;
    }

    public void loadUsers(ArrayList<_User> users){
        m_users = new ArrayList<>();
        for(_User u: users){
            m_users.add((_User4EUB) u);
        }
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
    public void EM() {
        initialize_probability(m_trainSet);

        int iter = 0;
        double lastAllLikelihood = 1.0;
        double currentAllLikelihood;
        double converge;
        do {
            currentAllLikelihood = E_step();

            if (iter >= 0)
                converge = Math.abs((lastAllLikelihood - currentAllLikelihood) / lastAllLikelihood);
            else
                converge = 1.0;

            calculate_M_step(iter);

            lastAllLikelihood = currentAllLikelihood;
        } while( ++iter < number_of_iteration && converge > m_converge);
    }

    @Override
    protected void initialize_probability(Collection<_Doc> collection) {

        System.out.println("[Info]Initializing topics, documents and users...");

        for(_Topic4EUB t: m_topics)
            t.setTopics4Variational(m_embedding_dim, t_mu, t_sigma);

        for(_Doc d: m_trainSet)
            ((_Doc4EUB) d).setTopics4Variational(number_of_topics, d_alpha, d_mu, d_sigma);

        for(_User4EUB u: m_users)
            u.setTopics4Variational(m_embedding_dim, m_users.size(), u_mu, u_sigma);

        init();
        Arrays.fill(m_alpha, d_alpha);

        // initialize topic-word allocation, p(w|z)
        for(_Doc doc:docs)
            updateStats4Doc(doc);
        for(int u_idx:m_mapByUser.keySet())
            updateStats4User(m_users.get(u_idx));
        for(int i_idx : m_mapByItem.keySet())
            updateStats4Item(m_items.get(i_idx));

        calculate_M_step(0);

    }

    public void E_step(){

    }
}