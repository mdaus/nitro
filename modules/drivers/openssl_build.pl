#!/usr/bin/perl -w

#  Current working directory support
use Cwd;

my $platform = $ARGV[0];

if(!$platform)
{
    print "Usage: " . $0 . " <target_platform>";
    exit;
}

my $cwd = Cwd::getcwd();
my $Library = $cwd . "/../net.ssl/lib/" . $platform . "/libssl.a";
my $LibrarySO = $cwd . "/../net.ssl/lib/" . $platform . "/libssl.so";
my $Include = $cwd . "/../net.ssl/include/openssl/ssl.h";
#  Constants
my $Tar = "tar -xvf ";
my $OpenSSL_Dist = "openssl-0.9.8";
my $Conf = "./config --prefix=";
my $Remove_Src = "rm -rf " . $OpenSSL_Dist . "/";
my $Make = "make";
my $Test = "make test";
my $Install = "make install";
#  Subroutines
sub do_cmd($) {
    my $cmd = shift;
    print "$0: " . $cmd . "\n";
    system $cmd;
}

if ( (-e $Include) && ((-e $Library) || (-e $LibrarySO))) {
	print "Found existing openssl driver: skipping installation.\n";
	exit;
}


#$ENV{'CFLAGS'} = join " ", @ARGV;
#print "CFLAGS=" . $ENV{ 'CFLAGS' } . "\n";


my $cmd = $Tar . $OpenSSL_Dist . ".tar";

#  Execute the command
do_cmd $cmd;

$cwd = Cwd::getcwd();
my $openssl_dir = $cwd . "/" . $OpenSSL_Dist;
chdir $openssl_dir;

$cmd = "mkdir -p " . $cwd . "/../net.ssl/lib/" . $platform;

do_cmd $cmd;

$cmd = $Conf . $cwd . "/openssl";

do_cmd $cmd;

#  Do make
do_cmd $Make;

# Do make test
do_cmd $Test;

#  Install openssl into ./openssl
do_cmd $Install;

$cmd = "mv " . $cwd . "/openssl/include/* " . $cwd . "/../net.ssl/include/";

do_cmd $cmd;

$cmd = "mv " . $cwd . "/openssl/lib/* " . $cwd . "/../net.ssl/lib/" . $platform . "/";

do_cmd $cmd;

chdir $cwd;

do_cmd $Remove_Src;

