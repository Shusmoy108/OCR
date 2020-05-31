/**
 * Created by Shusmoy on 12/20/2017.
 */
public class word {
    String w;
    public int c;
    int index;

    public int getIndex() {
        return index;
    }

    public void setW(String w) {

        this.w = w;
    }

    public void setC(int c) {
        this.c = c;
    }

    public void setIndex(int index) {
        this.index = index;
    }

    public word(String w) {
        this.w=w;
        c=0;
    }

    public String getW() {
        return w;
    }

    public int getC() {

        return c;
    }

    public boolean contain(String a)
    {
        if(w.equals(a))
        {
            return true;
        }
        return false;

    }
}
