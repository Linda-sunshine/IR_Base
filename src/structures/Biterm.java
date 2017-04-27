package structures;

import java.util.HashMap;
import java.util.Random;

/**
 * Created by jetcai1900 on 4/27/17.
 */
public class Biterm {
    int m_tid;
    _Word[] m_words;
    Random m_rand;

    public Biterm(int wid1, int wid2){
        createSpace(wid1, wid2);
    }

    protected void createSpace(int wid1, int wid2){
        if (m_words==null || m_words.length != 2) {
            m_words = new _Word[2];
        }

        _Word word1 = new _Word(wid1);
        m_words[0] = word1;

        _Word word2 = new _Word(wid2);
        m_words[1] = word2;

        if (m_rand==null)
            m_rand = new Random();
    }

    public void setTopics4GibbsbyRawToken(int numberOfTopics){
        int tid;

        tid = m_rand.nextInt(numberOfTopics);
        m_tid = tid;

        for(_Word w:m_words){
            w.setTopic(tid);
        }
    }

    public _Word[] getWords(){
        return m_words;
    }

    public int getTopic(){
        return m_tid;
    }

    public void setTopic(int tid){
        m_tid = tid;
    }
}
