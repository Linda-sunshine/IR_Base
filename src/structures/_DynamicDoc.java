package structures;

/**
 * Created by jetcai1900 on 3/2/17.
 */
public class _DynamicDoc extends _Doc{
    String m_timeStamp;

    public _DynamicDoc(int ID, String source, int ylabel, String timeStamp){
        super(ID, source, ylabel);
        m_timeStamp = timeStamp;
    }

    public String getDynamicTimeStamp(){
        return m_timeStamp;
    }
}
