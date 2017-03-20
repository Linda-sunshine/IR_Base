package structures;

/**
 * Created by jetcai1900 on 3/18/17.
 *
 * add x value to the word
 */
public class _Word4Embedding extends _Word {

    int m_xVal;

    public _Word4Embedding(int index){
        super(index);

    }

    public void setXVal(int xVal){
        m_xVal = xVal;
    }

    public int getXVal(){
        return m_xVal;
    }

}
