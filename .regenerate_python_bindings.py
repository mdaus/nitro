from subprocess import check_output, check_call, DEVNULL
import sys
from sys import stdout
from threading import Timer, Event
import itertools, sys


def thinking(evt):
    def thought():
        spinner = itertools.cycle(["-", "/", "|", "\\"])
        while not evt.isSet():
            stdout.write(next(spinner))  # write the next character
            stdout.flush()  # flush stdout buffer (actual character display)
            stdout.write("\b")  # erase the last written char

    return thought


def swig_version():
    decoded_result = check_output(["swig", "-version"]).decode("ascii")
    return decoded_result.split("\n")[1].split(" ")[-1]


def python(args):
    return check_call(
        [sys.executable] + args.split(" "), stderr=DEVNULL, stdout=DEVNULL
    )


def main():
    sv = swig_version()
    if sv != "3.0.12":
        print("Found unapproved swig version '{}'. Output may vary a great deal!".format(sv))

    print("Thinking about how great cmake is...")

    evt = Event()
    thread = Timer(0.7, thinking(evt))
    thread.start()

    try:
        python("waf configure --enable-debugging --prefix=installed --enable-swig")
        print("Just")

        python("waf build")
        print("about")

    except Exception as e:
        print("Thoughts interrupted by: ")
        print(e)
    finally:
        python("waf distclean")
        print("done")

    evt.set()


if __name__ == "__main__":
    main()
