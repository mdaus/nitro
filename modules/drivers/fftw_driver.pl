use Driver;

@ARGV >= 1 or die "We at least need the platform name!";

my $name = "fftw";
my $version = "2.1.5";
my %params = (
    Module => "fft",
    ShortName => $name,
    InstallRoot => shift,
    Where => shift() . "/" . $name . "-" . $version,
    Platform => shift,
    Header => $name . ".h",
    DriverDist => $name . "-" . $version,
    UnpackMethod => "tar",
    CFlags => shift,
    ConfOptions => "--enable-float"
);


my $driver = Driver::new(%params);


$driver->run;
