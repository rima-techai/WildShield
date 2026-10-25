// WildShield concept enclosure. Units: mm. NOT fabrication-ready.
// Component dimensions, optical fields of view and thermal clearances must be verified.
$fn=64;
part="assembly"; // assembly, body, lid, bracket
width=120; height=60; depth=40; wall=3; corner=6;
rgb_x=-30; thermal_x=30; rgb_d=20; thermal_d=24;
module rounded_box(w,h,d,r){
 linear_extrude(height=d) offset(r=r) square([w-2*r,h-2*r],center=true);
}
// Front is z=0. Front face points toward negative z.
module body(){
 difference(){
  rounded_box(width,height,depth,corner);
  translate([0,0,wall]) rounded_box(width-2*wall,height-2*wall,depth,corner-wall);
  for(spec=[[rgb_x,rgb_d],[thermal_x,thermal_d]])
   translate([spec[0],3,-1]) cylinder(h=wall+2,d=spec[1]);
  translate([0,-18,-1]) cylinder(h=wall+2,d=3); // status LED
  // Generic side cable opening, not a specified connector footprint.
  translate([width/2-wall-1,-6,depth-14]) cube([wall+2,12,6]);
 }
 // Internal prototype screw bosses; matching pilot holes in lid.
 for(x=[-width/2+9,width/2-9], y=[-height/2+9,height/2-9])
  translate([x,y,wall]) difference(){cylinder(h=depth-wall-2,d=7);translate([0,0,-1])cylinder(h=depth,d=2.5);}
}
module lid(){
 difference(){
 rounded_box(width,height,wall,corner);
 for(x=[-width/2+9,width/2-9],y=[-height/2+9,height/2-9]) translate([x,y,-1]) cylinder(h=wall+2,d=3.2);
 }
}
module bracket(){
 // Concept stand: base plus two side arms. Pivot bosses require mechanical detailing.
 difference(){
  union(){
   translate([-72,-45,-12]) cube([144,5,62]);
   for(x=[-72,66]) translate([x,-40,8]) cube([6,50,24]);
  }
  for(x=[-73,65]) translate([x,0,20]) rotate([0,90,0]) cylinder(h=9,d=5.5);
  for(x=[-45,45]) translate([x,-46,18]) rotate([-90,0,0]) cylinder(h=8,d=5.5);
 }
}
if(part=="body") body();
if(part=="lid") lid();
if(part=="bracket") bracket();
if(part=="assembly"){
 color([.17,.19,.19]) body();
 color([.25,.27,.27]) translate([0,0,depth+6]) lid();
 color([.3,.32,.32]) bracket();
 // Open apertures: no ordinary glass covering thermal sensor.
}
