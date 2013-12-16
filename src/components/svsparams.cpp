#include "svsparams.h"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>

#include <string>

using namespace boost::property_tree;

#define PUT(pt, f) pt.put(#f, f)
#define GET(pt, t, f) f = pt.get<t>(#f)

namespace {
    void initPTree(SVSParams const& ps, ptree * pt) {
        PUT((*pt), ps.Seed);
        PUT((*pt), ps.MaxAlpha);
        PUT((*pt), ps.SupportSize);
        PUT((*pt), ps.KernelThreshold);
        PUT((*pt), ps.TerminateEps);
        PUT((*pt), ps.SmoothingRange);
        PUT((*pt), ps.StepWidth);
        PUT((*pt), ps.TakeProb);
        PUT((*pt), ps.NumFP);
        PUT((*pt), ps.FPSpace);
        PUT((*pt), ps.CacheSize);
        PUT((*pt), ps.UseGrid);
        PUT((*pt), ps.UseNormals);
        PUT((*pt), ps.DoNormalizeGradient);
    }
}

void SVSParams::Load(const char* path) {
    ptree pt;
    read_xml(path, pt);
    GET(pt, int, Seed);
    GET(pt, float, MaxAlpha);
    GET(pt, float, SupportSize);
    GET(pt, float, KernelThreshold);
    GET(pt, float, TerminateEps);
    GET(pt, float, SmoothingRange);
    GET(pt, float,  StepWidth);
    GET(pt, float, TakeProb);
    GET(pt, int, NumFP);
    GET(pt, float, FPSpace);
    GET(pt, size_t, CacheSize);
    GET(pt, bool, UseGrid);
    GET(pt, bool, UseNormals);
    GET(pt, bool, DoNormalizeGradient);
}

void SVSParams::Save(const char* path) {
    ptree pt;
    initPTree(*this, &pt);
    xml_writer_settings<char> settings(' ', 4);
    write_xml(path, pt, std::locale(), settings);
}

std::string SVSParams::ToString() {
    ptree pt;
    initPTree(*this, &pt);
    xml_writer_settings<char> settings(' ', 4);
    std::stringstream sstr;
    write_xml(sstr, pt, settings);
    return sstr.str();
}

