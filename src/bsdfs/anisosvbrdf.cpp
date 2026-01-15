#include <mitsuba/core/fwd.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/string.h>
#include <mitsuba/core/timer.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/ior.h>
#include <mitsuba/render/microfacet.h>
#include <mitsuba/render/texture.h>
#include <drjit/dynamic.h>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class AnisoCookTorrance final : public BSDF<Float, Spectrum> {
public:
    MI_IMPORT_BASE(BSDF, m_flags, m_components)
    MI_IMPORT_TYPES(Texture)

    AnisoCookTorrance(const Properties &props) : Base(props) {

        m_eta = 1.5f;

        if (props.has_property("diffuse_reflectance"))
            m_diffuse_reflectance = props.get_texture<Texture>("diffuse_reflectance", 0.f);

        if (props.has_property("roughness"))
            m_roughness = props.get_texture<Texture>("roughness", 0.01f);

        m_normalmap = props.get_texture<Texture>("normalmap");
        m_tangentmap = props.get_texture<Texture>("tangentmap");

        if (props.has_property("diffuse_reflectance"))
            m_diffuse_reflectance = props.get_texture<Texture>("diffuse_reflectance", 0.f);
        if (props.has_property("specular_reflectance"))
            m_specular_reflectance = props.get_texture<Texture>("specular_reflectance", 1.f);

        m_components.push_back(BSDFFlags::GlossyReflection | BSDFFlags::FrontSide);
        m_components.push_back(BSDFFlags::DiffuseReflection | BSDFFlags::FrontSide);

        m_flags = m_components[0] | m_components[1] | BSDFFlags::Anisotropic;

        parameters_changed();
    }

    void traverse(TraversalCallback *callback) override {
        if (m_diffuse_reflectance)
            callback->put("diffuse_reflectance", m_diffuse_reflectance, ParamFlags::Differentiable);

        if (m_roughness)
            callback->put("roughness", m_roughness, ParamFlags::Differentiable);

        if (m_specular_reflectance)
            callback->put("specular_reflectance", m_specular_reflectance, ParamFlags::Differentiable);

        callback->put("normalmap", m_normalmap, ParamFlags::Differentiable | ParamFlags::Discontinuous);

        callback->put("tangentmap", m_tangentmap, ParamFlags::Differentiable | ParamFlags::Discontinuous);
    }

    void
    parameters_changed(const std::vector<std::string> &keys = {}) override {
        /* Compute weights that further steer samples towards
           the specular or diffuse components */
        Float d_mean = 0.5f, s_mean = 1.f;
        if (m_diffuse_reflectance)
            d_mean = m_diffuse_reflectance->mean();
        if (m_specular_reflectance)
            s_mean = m_specular_reflectance->mean();

        m_specular_sampling_weight = s_mean / (d_mean + s_mean);
    };

    std::pair<BSDFSample3f, Spectrum> sample(const BSDFContext &ctx,
                                             const SurfaceInteraction3f &si,
                                             Float sample1,
                                             const Point2f & sample2,
                                             Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFSample, active);

        // Sample nested BSDF with perturbed shading frame
        SurfaceInteraction3f perturbed_si(si);
        perturbed_si.sh_frame = frame(si, active);
        perturbed_si.wi = perturbed_si.to_local(si.wi);

        // Determine whether to sample the specular
        bool has_specular = ctx.is_enabled(BSDFFlags::GlossyReflection, 0),
             has_diffuse  = ctx.is_enabled(BSDFFlags::DiffuseReflection, 1);

        Float cos_theta_i = Frame3f::cos_theta(perturbed_si.wi);
        active &= cos_theta_i > 0.f;

        BSDFSample3f bs = dr::zeros<BSDFSample3f>();
        Spectrum result(0.f);
        if (unlikely((!has_specular && !has_diffuse) || dr::none_or<false>(active)))
            return { bs, result };

        Float prob_specular = m_specular_sampling_weight;
        if (unlikely(has_specular != has_diffuse))
            prob_specular = has_specular ? 1.f : 0.f;

        Mask sample_specular = active && (sample1 < prob_specular),
             sample_diffuse = active && !sample_specular;

        bs.eta = 1.f;

        if (dr::any_or<true>(sample_specular)) {
            Normal3f m = std::get<0>(sample_distr_specular(perturbed_si, sample2, active));

            dr::masked(bs.wo, sample_specular) = reflect(perturbed_si.wi, m);
            dr::masked(bs.sampled_component, sample_specular) = 0;
            dr::masked(bs.sampled_type, sample_specular) = +BSDFFlags::GlossyReflection;
        }

        if (dr::any_or<true>(sample_diffuse)) {
            dr::masked(bs.wo, sample_diffuse) = warp::square_to_cosine_hemisphere(sample2);
            dr::masked(bs.sampled_component, sample_diffuse) = 1;
            dr::masked(bs.sampled_type, sample_diffuse) = +BSDFFlags::DiffuseReflection;
        }

        bs.pdf = pdf(ctx, perturbed_si, bs.wo, active);
        active &= bs.pdf > 0.f;
        result = eval(ctx, perturbed_si, bs.wo, active);
        auto weight = (depolarizer<Spectrum>(result) / bs.pdf) & active;

        active &= dr::any(dr::neq(unpolarized_spectrum(weight), 0.f));
        // Transform sampled 'wo' back to original frame and check orientation
        Vector3f perturbed_wo = perturbed_si.to_world(bs.wo);
        active &= Frame3f::cos_theta(bs.wo) *
                      Frame3f::cos_theta(perturbed_wo) > 0.f;
        bs.wo = perturbed_wo;

        return { bs, weight & active };
    }

    Spectrum eval(const BSDFContext &ctx,
                  const SurfaceInteraction3f &si,
                  const Vector3f &wo, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        bool has_specular = ctx.is_enabled(BSDFFlags::GlossyReflection, 0),
             has_diffuse  = ctx.is_enabled(BSDFFlags::DiffuseReflection, 1);

        // Evaluate nested BSDF with perturbed shading frame
        SurfaceInteraction3f perturbed_si(si);
        perturbed_si.sh_frame = frame(si, active);
        perturbed_si.wi       = perturbed_si.to_local(si.wi);
        Vector3f perturbed_wo = perturbed_si.to_local(wo);

        active &= (Frame3f::cos_theta(wo) * Frame3f::cos_theta(perturbed_wo) > 0.f);

        Float cos_theta_i = Frame3f::cos_theta(perturbed_si.wi),
              cos_theta_o = Frame3f::cos_theta(perturbed_wo);

        active &= cos_theta_i > 0.f && cos_theta_o > 0.f;

        if (unlikely((!has_specular && !has_diffuse) || dr::none_or<false>(active)))
            return 0.f;

        UnpolarizedSpectrum value(0.f);
        if (has_specular) {
            // Calculate the reflection half-vector
            Vector3f H = dr::normalize(perturbed_wo + perturbed_si.wi);
            Float alpha_u, alpha_v;
            std::tie(alpha_u, alpha_v) = eval_roughness(perturbed_si, active);

            // Evaluate the microfacet normal distribution
            Float D = eval_microfacet_distr(H, alpha_u, alpha_v);

            // Fresnel term
            Float F = std::get<0>(fresnel(dr::dot(perturbed_si.wi, H), Float(m_eta)));

            // Smith's shadow-masking function
            Float G = smith_g1(perturbed_si.wi, H, alpha_u, alpha_v) *
                      smith_g1(perturbed_wo, H, alpha_u, alpha_v);

            // Calculate the specular reflection component
            value = F * D * G / (4.f * cos_theta_i);

            if (m_specular_reflectance)
                value *= m_specular_reflectance->eval(perturbed_si, active);
        }

        if (has_diffuse) {
            value += m_diffuse_reflectance->eval(perturbed_si, active) * dr::InvPi<Float> * cos_theta_o;
        }

        return depolarizer<Spectrum>(value) & active;
    }

    Float pdf(const BSDFContext &ctx,
              const SurfaceInteraction3f &si, const Vector3f &wo,
              Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        bool has_specular = ctx.is_enabled(BSDFFlags::GlossyReflection, 0),
             has_diffuse = ctx.is_enabled(BSDFFlags::DiffuseReflection, 1);

        // Evaluate nested BSDF with perturbed shading frame
        SurfaceInteraction3f perturbed_si(si);
        perturbed_si.sh_frame = frame(si, active);
        perturbed_si.wi       = perturbed_si.to_local(si.wi);
        Vector3f perturbed_wo = perturbed_si.to_local(wo);

        active &= Frame3f::cos_theta(wo) *
                      Frame3f::cos_theta(perturbed_wo) > 0.f;

        Float cos_theta_i = Frame3f::cos_theta(perturbed_si.wi),
              cos_theta_o = Frame3f::cos_theta(perturbed_wo);

        active &= cos_theta_i > 0.f && cos_theta_o > 0.f;

        if (unlikely((!has_specular && !has_diffuse) || dr::none_or<false>(active)))
            return 0.f;

        Float prob_diffuse = 1.f - m_specular_sampling_weight,
              prob_specular = m_specular_sampling_weight;
        if (unlikely(has_specular != has_diffuse))
            prob_specular = has_specular ? 1.f : 0.f;
        prob_diffuse = 1.f - prob_specular;

        Float result = 0.f;


        if (has_specular) {
            Float alpha_u, alpha_v;
            std::tie(alpha_u, alpha_v) = eval_roughness(perturbed_si, active);
            Vector3f H  = dr::normalize(perturbed_wo + perturbed_si.wi);
            result = eval_microfacet_distr(H, alpha_u, alpha_v) *
                     smith_g1(perturbed_si.wi, H, alpha_u, alpha_v) /
                     (4.f * cos_theta_i);
        }
        result *= prob_specular;

        if (has_diffuse)
            result += prob_diffuse * warp::square_to_cosine_hemisphere_pdf(perturbed_wo);

        return result;
    }

    std::pair<Spectrum, Float> eval_pdf(const BSDFContext &ctx,
                                        const SurfaceInteraction3f &si,
                                        const Vector3f &wo,
                                        Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        bool has_specular = ctx.is_enabled(BSDFFlags::GlossyReflection, 0),
             has_diffuse  = ctx.is_enabled(BSDFFlags::DiffuseReflection, 1);

        // Evaluate nested BSDF with perturbed shading frame
        SurfaceInteraction3f perturbed_si(si);
        perturbed_si.sh_frame = frame(si, active);
        perturbed_si.wi       = perturbed_si.to_local(si.wi);
        Vector3f perturbed_wo = perturbed_si.to_local(wo);

        active &= Frame3f::cos_theta(wo) *
                      Frame3f::cos_theta(perturbed_wo) > 0.f;

        Float cos_theta_i = Frame3f::cos_theta(perturbed_si.wi),
              cos_theta_o = Frame3f::cos_theta(perturbed_wo);

        active &= cos_theta_i > 0.f && cos_theta_o > 0.f;

        if (unlikely((!has_specular && !has_diffuse) || dr::none_or<false>(active)))
            return { 0.f, 0.f };

        Float prob_diffuse = 1.f - m_specular_sampling_weight,
              prob_specular = m_specular_sampling_weight;
        if (unlikely(has_specular != has_diffuse))
            prob_specular = has_specular ? 1.f : 0.f;
        prob_diffuse = 1.f - prob_specular;

        Float pdf = 0.f;
        UnpolarizedSpectrum value(0.f);

        if (has_specular) {
            Vector3f H  = dr::normalize(perturbed_wo + perturbed_si.wi);
            Float alpha_u, alpha_v;
            std::tie(alpha_u, alpha_v) = eval_roughness(perturbed_si, active);

            // Evaluate the microfacet normal distribution
            Float D = eval_microfacet_distr(H, alpha_u, alpha_v);

            // Evaluate shadow/masking term for incoming direction
            Float smith_g1_wi = smith_g1(perturbed_si.wi, H, alpha_u, alpha_v);

            pdf = D * smith_g1_wi / (4.f * cos_theta_i) * prob_specular;

            // Fresnel term
            Float F = std::get<0>(fresnel(dr::dot(perturbed_si.wi, H), Float(m_eta)));

            // Smith's shadow-masking function
            Float G = smith_g1(perturbed_wo, H, alpha_u, alpha_v) * smith_g1_wi;

            // Calculate the specular reflection component
            value = F * D * G / (4.f * cos_theta_i);

            if (m_specular_reflectance)
                value *= m_specular_reflectance->eval(perturbed_si, active);
        }

        if (has_diffuse) {
            pdf += prob_diffuse * warp::square_to_cosine_hemisphere_pdf(perturbed_wo);

            value +=
                m_diffuse_reflectance->eval(perturbed_si, active) *
                dr::InvPi<Float> * cos_theta_o;
        }

        return { depolarizer<Spectrum>(value) & active, pdf };
    }

    Frame3f frame(const SurfaceInteraction3f &si, Mask active) const {
        Normal3f n = dr::fmadd(m_normalmap->eval_3(si, active), 2, -1.f);
        Normal3f t = dr::fmadd(m_tangentmap->eval_3(si, active), 2, -1.f);

        Frame3f result;
        result.n = dr::normalize(n);
        result.t = dr::normalize(t);
        result.s = dr::cross(result.t, result.n);
        return result;
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "AnisoCookTorrance[" << std::endl;
        if (m_diffuse_reflectance)
            oss << "  diffuse_reflectance = "      << m_diffuse_reflectance               << "," << std::endl;

        if (m_roughness)
            oss << "  roughness = "      << m_roughness               << "," << std::endl;

        if (m_specular_reflectance)
            oss << "  specular_reflectance = "     << m_specular_reflectance              << "," << std::endl;

        if (m_normalmap)
            oss << "  normalmap = " << string::indent(m_normalmap) << "," << std::endl;

        if (m_tangentmap)
            oss << "  tangentmap = " << string::indent(m_tangentmap) << "," << std::endl;

        oss << "  specular_sampling_weight = " << m_specular_sampling_weight          << "," << std::endl
            << "  eta = "                      << m_eta                               << "," << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS(AnisoCookTorrance)

private:
    ref<Texture> m_normalmap;
    ref<Texture> m_tangentmap;
    ref<Texture> m_diffuse_reflectance;
    ref<Texture> m_roughness;
    ref<Texture> m_specular_reflectance;
    Float m_eta;
    Float m_specular_sampling_weight;

    std::pair<Normal3f, Float>  sample_distr_specular(const SurfaceInteraction3f &si,
                                                      const Point2f & sample,
                                                      Mask active) const {
        // Visible normal sampling.
        Float sin_phi, cos_phi, cos_theta;
        Float alpha_u, alpha_v;

        // Step 1: stretch wi
        std::tie(alpha_u, alpha_v) = eval_roughness(si, active);
        Vector3f wi_p = dr::normalize(Vector3f(
            alpha_u * si.wi.x(),
            alpha_v * si.wi.y(),
            si.wi.z()
            ));

        std::tie(sin_phi, cos_phi) = Frame3f::sincos_phi(wi_p);
        cos_theta = Frame3f::cos_theta(wi_p);

        // Step 2: simulate P22_{wi}(slope.x, slope.y, 1, 1)
        // Choose a projection direction and re-scale the sample
        Point2f p = warp::square_to_uniform_disk_concentric(sample);

        Float s = 0.5f * (1.f + cos_theta);
        p.y() = dr::lerp(dr::safe_sqrt(1.f - dr::sqr(p.x())), p.y(), s);

        // Project onto chosen side of the hemisphere
        Float x = p.x(), y = p.y(),
              z = dr::safe_sqrt(1.f - dr::squared_norm(p));

        // Convert to slope
        Float sin_theta_i = dr::safe_sqrt(1.f - dr::sqr(cos_theta));
        Float norm = dr::rcp(dr::fmadd(sin_theta_i, y, cos_theta * z));
        Vector2f slope = Vector2f(dr::fmsub(cos_theta, y, sin_theta_i * z), x) * norm;

        // Step 3: rotate & unstretch
        slope = Vector2f(
            dr::fmsub(cos_phi, slope.x(), sin_phi * slope.y()) * alpha_u,
            dr::fmadd(sin_phi, slope.x(), cos_phi * slope.y()) * alpha_v);

        // Step 4: compute normal & PDF
        Normal3f m = dr::normalize(Vector3f(-slope.x(), -slope.y(), 1));

        Float pdf = eval_microfacet_distr(m, alpha_u, alpha_v) * smith_g1(si.wi, m, alpha_u, alpha_v)
                    * dr::abs_dot(si.wi, m) / Frame3f::cos_theta(si.wi);

        return { m, pdf };
    }

    Float eval_microfacet_distr(const Vector3f &m, const Float alpha_u, const Float alpha_v) const {
        Float alpha_uv = alpha_u * alpha_v,
              cos_theta         = Frame3f::cos_theta(m),
              cos_theta_2       = dr::sqr(cos_theta),
              result;

        // GGX / Trowbridge-Reitz distribution function
        result =
            dr::rcp(dr::Pi<Float> * alpha_uv *
                    dr::sqr(dr::sqr(m.x() / alpha_u) +
                            dr::sqr(m.y() / alpha_v) + dr::sqr(m.z())));

        // Prevent potential numerical issues in other stages of the model
        return dr::select(result * cos_theta > 1e-20f, result, 0.f);
    }

    Float smith_g1(const Vector3f &v, const Vector3f &m, const Float alpha_u, const Float alpha_v) const {
        Float xy_alpha_2 = dr::sqr(alpha_u * v.x()) + dr::sqr(alpha_v * v.y()),
              tan_theta_alpha_2 = xy_alpha_2 / dr::sqr(v.z()),
              result;

        result = 2.f / (1.f + dr::sqrt(1.f + tan_theta_alpha_2));

        // Perpendicular incidence -- no shadowing/masking
        dr::masked(result, dr::eq(xy_alpha_2, 0.f)) = 1.f;

        /* Ensure consistent orientation (can't see the back
           of the microfacet from the front and vice versa) */
        dr::masked(result, dr::dot(v, m) * Frame3f::cos_theta(v) <= 0.f) = 0.f;

        return result;
    }

    inline std::pair<Float, Float> eval_roughness(const SurfaceInteraction3f &si, Mask active) const {
        Float alpha_u = dr::maximum(m_roughness->eval(si, active).x(), (Float) 1e-4f);
        Float alpha_v = dr::maximum(m_roughness->eval(si, active).y(), (Float) 1e-4f);
        return {alpha_u, alpha_v};
    }
};

MI_EXPORT_PLUGIN(AnisoCookTorrance);
NAMESPACE_END(mitsuba)

