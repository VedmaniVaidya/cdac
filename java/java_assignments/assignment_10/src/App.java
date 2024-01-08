import java.awt.*;

class MovingCircle extends Frame implements Runnable {
    private int x;
    private int direction;
    private Thread thread;

    public MovingCircle() {
        x = 100;
        direction = 1;
        thread = new Thread(this);
        thread.start();
    }

    public void paint(Graphics g) {
        g.setColor(Color.RED);
        g.fillOval(x, 100, 100, 100);
    }

    public void run() {
        while (true) {
            if (direction == 1) {
                x++;
                if (x == this.getWidth() - 100) {
                    direction = -1;
                }
            } else {
                x--;
                if (x == 0) {
                    direction = 1;
                }
            }
            try {
                Thread.sleep(20);
            } catch (Exception e) {
                e.printStackTrace();
            }
            repaint();
        }
    }

}


public class App {
    public static void main(String[] args) {
        MovingCircle mc = new MovingCircle();
        mc.setSize(500, 300);
        mc.setVisible(true);
    }
}
