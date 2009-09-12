use Driver;

@ARGV >= 1 or die "We at least need the platform name!";

my $name    = "e2fsprogs";
my $version = "1.40-uuid";
my %params = (
    Module => "unique",
    ShortName => "uuid",
    InstallRoot => shift,
    Where => shift() . "/$name-$version",
    Platform => shift,
    Header => "uuid/uuid.h",
    DriverDist => "$name-$version",
    UnpackMethod => "tar",
    CFlags => shift,
    ConfOptions => shift
);


my $driver = Driver::new(%params);


$driver->run;
