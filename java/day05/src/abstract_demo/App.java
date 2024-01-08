package abstract_demo;

public class App {
    public static void main(String[] args) {
        Shape c = new Circle(10);
        Shape r = new Rectangle(10,20);
        Shape s = new Square(10);
        
        System.out.println(c.CalcArea());
        System.out.println(r.CalcArea());
        System.out.println(s.CalcArea());
        System.out.println(c.CalcPerimeter());
        System.out.println(r.CalcPerimeter());
        System.out.println(s.CalcPerimeter());
    }
}


abstract class Shape{

    public abstract float CalcArea();
    public abstract float CalcPerimeter();

}


class Rectangle extends Shape{
    float length,breadth;

    public Rectangle() {
        super();
        this.length=0;
        this.breadth=0;
    }
    public Rectangle(float length,float breadth) {
        super();
        this.length = length;
        this.breadth = breadth;
    }

    @Override
    public float CalcArea() {
        float area=0;
        area = this.length*this.breadth;
        return area;
    }

    @Override
    public float CalcPerimeter() {
        float perimeter=0;
        perimeter = 2*(this.length+this.breadth);
        return perimeter;
    }

}


class Square extends Rectangle{
    float side;
    public Square() {
        super(0,0);
    }

    public Square(float side) {
        super(side,side);
    }

    @Override
    public float CalcArea() {
        return super.CalcArea();
    }

    @Override
    public float CalcPerimeter() {
        return super.CalcPerimeter();
    }
}


class Circle extends Shape{
    float radius;

    public Circle() {
        this.radius = 0;
    }    
    public Circle(float radius) {
        this.radius = radius;
    }

    @Override
    public float CalcArea() {
        float area=0;
        area = (22.0f/7)*radius*radius;
        return area;
    }

    @Override
    public float CalcPerimeter() {
        float perimeter=0;
        perimeter = 2.0f*(22.0f/7)*radius;
        return perimeter;
    }
    
}
