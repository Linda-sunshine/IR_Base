package topicmodels.correspondenceModels;

import structures.*;

import java.util.Arrays;
import java.util.Collection;

/**
 * Created by jetcai1900 on 4/7/17.
 */
public class LDAGibbs4ACWithRawToken extends LDAGibbs4AC{
    public LDAGibbs4ACWithRawToken(int number_of_iteration, double converge, double beta,
                                   _Corpus c, double lambda, int number_of_topics, double alpha,
                                   double burnIn, int lag){
        super(number_of_iteration,  converge,  beta, c, lambda, number_of_topics,  alpha,  burnIn,  lag);
    }

    public String toString() {
        return String
                .format("LDAGibbs4ACWithRawToken[k:%d, alphaA:%.2f, beta:%.2f, trainProportion:%.2f, Gibbs Sampling]",
                        number_of_topics, d_alpha, d_beta,
                        1 - m_testWord4PerplexityProportion);
    }

    protected void initialize_probability(Collection<_Doc> collection) {
        createSpace();
        for (int i = 0; i < number_of_topics; i++) {
            Arrays.fill(topic_term_probabilty[i], 0);
            Arrays.fill(word_topic_sstat[i], d_beta);
        }
        Arrays.fill(m_sstat, d_beta * vocabulary_size);

        for (_Doc d : collection) {
            if (d instanceof _ParentDoc) {
                for (_Stn stnObj : d.getSentences()) {
                    stnObj.setTopicsVct(number_of_topics);
                }
                ((_ParentDocWithRawToken)d).setTopics4GibbsbyRawToken(number_of_topics, d_alpha);
            } else if (d instanceof _ChildDoc) {
                d.setTopics4GibbsbyRawTokenLDA(number_of_topics, d_alpha);
            }

            for (_Word w : d.getWords()) {
                word_topic_sstat[w.getTopic()][w.getIndex()]++;
                m_sstat[w.getTopic()]++;
            }
        }

        imposePrior();
    }
}
