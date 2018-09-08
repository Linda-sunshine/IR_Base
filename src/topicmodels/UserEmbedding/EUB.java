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
    /***to be done***/
    protected void init() { // clear up for next iteration during EM
        super.init();
    }

    @Override
    protected void initialize_probability(Collection<_Doc> docs) {

        System.out.println("[Info]Initializing topics, documents and users...");

        for(_Topic4EUB t: m_topics)
            t.setTopics4Variational(m_embedding_dim, t_mu, t_sigma);

        for(_Doc d: m_trainSet)
            ((_Doc4EUB) d).setTopics4Variational(number_of_topics, d_alpha, d_mu, d_sigma);

        for(_User4EUB u: m_users)
            u.setTopics4Variational(m_embedding_dim, m_users.size(), u_mu, u_sigma);

        init();
        Arrays.fill(m_alpha, d_alpha);

        for(_Topic4EUB topic: m_topics)
            updateStats4Topic(topic);

        for(_Doc doc: m_trainSet)
            updateStats4Doc((_Doc4EUB) doc);

        for(_User4EUB user: m_users)
            updateStats4User(user);

        calculate_M_step(0);
    }

    /***to be done***/
    // Update the stats related with prior Gaussian distribution
    // \alpha
    public void updateStats4Topic(_Topic4EUB topic){

    }

    /***to be done***/
    // Update the stats related with doc
    // \tau and \beta
    public void updateStats4Doc(_Doc4EUB doc){

    }

    /***to be done***/
    // Update the stats related with prior Gaussian distribution
    // \gamma
    public void updateStats4User(_User4EUB user){

    }

    // update variational parameters of latent variables
    protected double E_step(){
        int iter = 0;
        double totalLikelihood, last = -1.0, converge;

        init();
        do {
            totalLikelihood = 0.0;
            for (_Doc doc: m_trainSet) {
                totalLikelihood += varInference4Doc((_Doc4EUB) doc);
            }
            for (_Topic4EUB topic: m_topics)
                totalLikelihood += varInference4Topic(topic);

            for(_User4EUB user: m_users)
                totalLikelihood += varInference4User(user);

            if(Double.isNaN(totalLikelihood) || Double.isInfinite(totalLikelihood))
                System.out.println("[error] The likelihood is Nan or Infinity!!");

            if(iter > 0)
                converge = Math.abs((totalLikelihood - last) / last);
            else
                converge = 1.0;

            last = totalLikelihood;

            if(iter % 10 == 0)
                System.out.format("[Info]Single-thread E-Step: %d iteration, likelihood=%.2f, converge to %.8f\n",
                        iter, last, converge);

        }while(iter++ < m_varMaxIter && converge > m_varConverge);

        //collect sufficient statistics for model parameter updates
        for(_Topic4EUB topic: m_topics)
            updateStats4Topic(topic);

        for(_Doc doc: m_trainSet)
            updateStats4Doc((_Doc4EUB) doc);

        for(_User4EUB user: m_users)
            updateStats4User(user);

        return totalLikelihood;
    }

    /***to be done***/
    protected double varInference4Topic(_Topic4EUB topic){


        return calc_log_likelihood_per_topic(topic);
    }

    /***to be done***/
    protected double varInference4Doc(_Doc4EUB doc){

        return calc_log_likelihood_per_doc(doc);
    }

    /***to be done***/
    protected double varInference4User(_User4EUB user){

        return calc_log_likelihood_per_user(user);
    }

    protected double calc_log_likelihood_per_topic(_Topic4EUB topic){
        return 0;
    }

    protected double calc_log_likelihood_per_doc(_Doc4EUB doc){
        return 0;
    }

    protected double calc_log_likelihood_per_user(_User4EUB user){
        return 0;
    }
}