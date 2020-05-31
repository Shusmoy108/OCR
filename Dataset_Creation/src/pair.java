

/**
 * Created by Shusmoy on 12/22/2017.
 */
public class pair {
    int p1;
    int p2;
    String pw1;
    String pw2;

    public int getP2() {
        return p2;
    }

    public String getPw1() {
        return pw1;
    }

    public String getPw2() {
        return pw2;
    }

    public pair(int p1, int p2, String pw1, String pw2) {

        this.p1 = p1;
        this.p2 = p2;
        this.pw1 = pw1;
        this.pw2 = pw2;
    }

    public int getP1() {

        return p1;
    }

    public pair(int p1, int p2) {
        this.p1 = p1;
        this.p2 = p2;
    }
}
