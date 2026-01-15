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
class CookTorrance final : public BSDF<Float, Spectrum> {
public:
    MI_IMPORT_BASE(BSDF, m_flags, m_components)
    MI_IMPORT_TYPES(Texture)

    CookTorrance(const Properties &props) : Base(props) {

        m_eta = 1.0f;

        if (props.has_property("diffuse_reflectance"))
            m_diffuse_reflectance = props.get_texture<Texture>("diffuse_reflectance", 0.f);
        if (props.has_property("roughness"))
            m_roughness = props.get_texture<Texture>("roughness", 0.01f);
        if (props.has_property("specular_reflectance"))
            m_specular_reflectance = props.get_texture<Texture>("specular_reflectance", 1.f);

        m_components.push_back(BSDFFlags::GlossyReflection | BSDFFlags::FrontSide);
        m_components.push_back(BSDFFlags::DiffuseReflection | BSDFFlags::FrontSide);

        m_flags = m_components[0] | m_components[1];
        // dr::set_attr(this, "flags", m_flags);

        parameters_changed();
    }

    void traverse(TraversalCallback *callback) override {
        if (m_diffuse_reflectance)
            callback->put("diffuse_reflectance",
                                 m_diffuse_reflectance.get(),
                                 +ParamFlags::Differentiable);
        if (m_roughness)
            callback->put("roughness",
                                 m_roughness.get(),
                                 +ParamFlags::Differentiable);
        if (m_specular_reflectance)
            callback->put("specular_reflectance",
                                 m_specular_reflectance.get(),
                                 +ParamFlags::Differentiable);
    }

    void parameters_changed(const std::vector<std::string> &keys = {}) override {
        /* Compute weights that further steer samples towards
           the specular or diffuse components */
        Float d_mean = m_diffuse_reflectance->mean(), s_mean = 1.f;

        if (m_specular_reflectance)
            s_mean = m_specular_reflectance->mean();

        m_specular_sampling_weight = s_mean / (d_mean + s_mean);
    };

    std::pair<BSDFSample3f, Spectrum> sample(const BSDFContext &ctx,
                                             const SurfaceInteraction3f &si,
                                             Float /* sample1 */,
                                             const Point2f & sample2,
                                             Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFSample, active);

        // Determine whether to sample the specular
        bool has_specular = ctx.is_enabled(BSDFFlags::GlossyReflection, 0),
             has_diffuse  = ctx.is_enabled(BSDFFlags::DiffuseReflection, 1);

        Float cos_theta_i = Frame3f::cos_theta(si.wi);
        active &= cos_theta_i > 0.f;

        BSDFSample3f bs = dr::zeros<BSDFSample3f>();
        Spectrum result(0.f);
        if (unlikely((!has_specular && !has_diffuse) || dr::none_or<false>(active)))
            return { bs, result };

        Float prob_specular;

        if (unlikely(has_specular != has_diffuse))
            prob_specular = has_specular ? 1.f : 0.f;
        else
            prob_specular =
                dr::select(sample2.x() <= m_specular_sampling_weight,
                           sample2.x() / m_specular_sampling_weight,
                           (sample2.x() - m_specular_sampling_weight) /
                               (1.f - m_specular_sampling_weight));

        Mask sample_specular = active && (prob_specular <= m_specular_sampling_weight),
             sample_diffuse = active && !sample_specular;
        Float new_sample2_x = prob_specular;
        Float new_sample2_y = sample2.y();
        // sample2.x() = prob_specular;

        bs.eta = 1.f;

        if (dr::any_or<true>(sample_specular)) {
            const Normal3f m = sample_specular_branch(si, new_sample2_x, new_sample2_y, sample_specular);
            dr::masked(bs.wo, sample_specular) = 2.f * dr::dot(si.wi, m) * Vector3f(m) - si.wi;
            dr::masked(bs.sampled_component, sample_specular) = 0;
            dr::masked(bs.sampled_type, sample_specular) = +BSDFFlags::GlossyReflection;
        }
        const Point2f new_sample2 = Point2f(new_sample2_x, new_sample2_y);
        if (dr::any_or<true>(sample_diffuse)) {
            dr::masked(bs.wo, sample_diffuse) = warp::square_to_cosine_hemisphere(new_sample2);
            dr::masked(bs.sampled_component, sample_diffuse) = 1;
            dr::masked(bs.sampled_type, sample_diffuse) = +BSDFFlags::DiffuseReflection;
        }

        active &= Frame3f::cos_theta(bs.wo) > 0.f;
        bs.pdf = pdf(ctx, si, bs.wo, active);
        active &= bs.pdf > 0.f;
        result = eval(ctx, si, bs.wo, active);
        return { bs, (depolarizer<Spectrum>(result) / bs.pdf) & active };
    }

    Spectrum eval(const BSDFContext &ctx,
                  const SurfaceInteraction3f &si,
                  const Vector3f &wo, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        bool has_specular = ctx.is_enabled(BSDFFlags::GlossyReflection, 0),
             has_diffuse  = ctx.is_enabled(BSDFFlags::DiffuseReflection, 1);

        Float cos_theta_i = Frame3f::cos_theta(si.wi),
              cos_theta_o = Frame3f::cos_theta(wo);

        active &= (cos_theta_i > 0.f) && (cos_theta_o > 0.f);

        if (unlikely((!has_specular && !has_diffuse) || dr::none_or<false>(active)))
            return 0.f;

        UnpolarizedSpectrum value(0.f);
        if (has_specular) {
            Vector3f H = dr::normalize(wo + si.wi);
            value +=
                dr::select(Frame3f::cos_theta(H) > 0.f,
                           eval_specular(H, cos_theta_i, si, wo, active), 0.f);
        }

        if (has_diffuse) {
            value += m_diffuse_reflectance->eval(si, active) * (Spectrum(1.f) - m_specular_reflectance->eval(si, active)) * dr::InvPi<Float> * Frame3f::cos_theta(wo);
        }
        return depolarizer<Spectrum>(value) & active;
    }

    Float pdf(const BSDFContext &ctx,
              const SurfaceInteraction3f &si, const Vector3f &wo,
              Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        bool has_specular = ctx.is_enabled(BSDFFlags::GlossyReflection, 0),
             has_diffuse = ctx.is_enabled(BSDFFlags::DiffuseReflection, 1);

        Float cos_theta_i = Frame3f::cos_theta(si.wi),
              cos_theta_o = Frame3f::cos_theta(wo);

        active &= cos_theta_i > 0.f && cos_theta_o > 0.f;

        if (unlikely((!has_specular && !has_diffuse) || dr::none_or<false>(active)))
            return 0.f;

        Float prob_diffuse = 0.0f, prob_specular = 0.0f;

        if (has_diffuse)
            prob_diffuse = warp::square_to_cosine_hemisphere_pdf(wo);

        if (has_specular) {
            Vector3f H  = wo + si.wi;
            Float H_len = dr::norm(H);
            H           = dr::normalize(H);
            prob_specular = dr::select(
                H_len > 0.f, specular_pdf(H, cos_theta_i, si, active), 0.f);
        }

        return m_specular_sampling_weight * prob_specular + (1.f - m_specular_sampling_weight) * prob_diffuse;
    }

    std::pair<Spectrum, Float> eval_pdf(const BSDFContext &ctx,
                                        const SurfaceInteraction3f &si,
                                        const Vector3f &wo,
                                        Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        bool has_specular = ctx.is_enabled(BSDFFlags::GlossyReflection, 0),
             has_diffuse  = ctx.is_enabled(BSDFFlags::DiffuseReflection, 1);

        Float cos_theta_i = Frame3f::cos_theta(si.wi),
              cos_theta_o = Frame3f::cos_theta(wo);

        active &= cos_theta_i > 0.f && cos_theta_o > 0.f;

        if (unlikely((!has_specular && !has_diffuse) || dr::none_or<false>(active)))
            return { 0.f, 0.f };

        Float prob_diffuse = 0.0f, prob_specular = 0.0f;
        Float pdf = 0.f;
        UnpolarizedSpectrum value(0.f);

        if (has_diffuse) {
            prob_diffuse = warp::square_to_cosine_hemisphere_pdf(wo);
            pdf += (1.f - m_specular_sampling_weight) * prob_diffuse;

            value +=
                m_diffuse_reflectance->eval(si, active) *
                (UnpolarizedSpectrum(1.f) - m_specular_reflectance->eval(si, active)) *
                dr::InvPi<Float> * Frame3f::cos_theta(wo);
        }

        if (has_specular) {
            Vector3f H  = wo + si.wi;
            Float H_len = dr::norm(H);
            H           = dr::normalize(H);
            prob_specular = dr::select(
                H_len > 0.f, specular_pdf(H, cos_theta_i, si, active), 0.f);
            pdf += m_specular_sampling_weight * prob_specular;
            value +=
                dr::select(Frame3f::cos_theta(H) > 0.f,
                           eval_specular(H, cos_theta_i, si, wo, active), 0.f);
        }

        return { depolarizer<Spectrum>(value) & active, pdf };
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "CookTorrance[" << std::endl;
        if (m_diffuse_reflectance)
            oss << "  diffuse_reflectance = "      << m_diffuse_reflectance               << "," << std::endl;

        if (m_roughness)
            oss << "  roughness = "      << m_roughness               << "," << std::endl;

        if (m_specular_reflectance)
            oss << "  specular_reflectance = "     << m_specular_reflectance              << "," << std::endl;

        oss << "  specular_sampling_weight = " << m_specular_sampling_weight          << "," << std::endl
            << "  eta = "                      << m_eta                               << "," << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS(CookTorrance)
private:
    ref<Texture> m_diffuse_reflectance;
    ref<Texture> m_roughness;
    ref<Texture> m_specular_reflectance;
    Float m_eta;
    Float m_specular_sampling_weight;

    const Normal3f sample_specular_branch(const SurfaceInteraction3f &si, Float & sample2_x, Float & sample2_y, Mask active) const {
        Float phi_m = (2.f * dr::Pi<Float>) * sample2_y;
        Float roughness = dr::maximum(m_roughness->eval(si, active).x(), (Float)1e-4f);
        const Float alpha = roughness * roughness;
        Float sin_phi_m, cos_phi_m;
        std::tie(sin_phi_m, cos_phi_m) = dr::sincos(phi_m);

        Float theta =
            dr::select(si.wi.z() < (Float) 0.9999, dr::acos(si.wi.z()), 0);
        Float r = dr::safe_sqrt(sample2_x / (1 - sample2_x));
        Vector2f slope =
            dr::select(theta < 1e-4f, Vector2f(r * cos_phi_m, r * sin_phi_m),
                       sample_specular_slope(theta, sample2_x, sample2_y));
        slope = Vector2f(cos_phi_m * slope.x() - sin_phi_m * slope.y(),
                                  sin_phi_m * slope.x() + cos_phi_m * slope.y());
        slope *= alpha;

        Float normalization = (Float) 1.f / dr::sqrt(slope.x() * slope.x() + slope.y() * slope.y() + (Float) 1.f);
        Normal3f m = Normal3f(-slope.x() * normalization, -slope.y() * normalization, normalization);
        return m;
    }

    Vector2f sample_specular_slope(Float theta, Float &sample2_x, Float &sample2_y) const {
        Float tan_theta = dr::tan(theta);
        Float a = 1.f / tan_theta;
        Float G1 = 2.f / (1.f + dr::safe_sqrt(1.f + 1.f / (a * a)));

        Float A = 2.f * sample2_x / G1 - 1.f;
        A               = dr::select(dr::abs(A) >= 1.f,
                                     dr::select(A >= 0.f, A - 0.0001f, A + 0.0001f), A);
        Float tmp = 1.f / (A * A - 1.f);
        Float B = tan_theta;
        Float D = dr::safe_sqrt(B * B * tmp * tmp - (A * A - B * B) * tmp);
        Float slope_x_1 = B * tmp - D;
        Float slope_x_2 = B * tmp + D;
        Float slope_x = dr::select(A < 0.f || slope_x_2 > 1.f / tan_theta, slope_x_1, slope_x_2);

        Float S   = dr::select(sample2_y > 0.5f, 1.f, -1.f);
        sample2_y = dr::select(sample2_y > 0.5f, 2.f * (sample2_y - 0.5f),
                               2.f * (0.5f - sample2_y));
        Float z =
            (sample2_y *
                 (sample2_y * (sample2_y * (-(Float) 0.365728915865723) +
                               (Float) 0.790235037209296) -
                  (Float) 0.424965825137544) +
             (Float) 0.000152998850436920) /
            (sample2_y *
                 (sample2_y *
                      (sample2_y * (sample2_y * (Float) 0.169507819808272 -
                                    (Float) 0.397203533833404) -
                       (Float) 0.232500544458471) +
                  (Float) 1) -
             (Float) 0.539825872510702);
        Float slope_y = S * z * dr::sqrt(1.f + slope_x * slope_x);
        Vector2f slope = Vector2f(slope_x, slope_y);
        return slope;
    }

    UnpolarizedSpectrum eval_specular(Vector3f &H, Float cos_theta_i,
                                      const SurfaceInteraction3f &si,
                           const Vector3f &wo, Mask active) const {
        Float roughness = dr::maximum(m_roughness->eval(si, active).x(), (Float) 1e-4f);
        const Float alpha = roughness * roughness;
        const Float alpha2 = alpha * alpha;
        const Float cos_theta2 = Frame3f::cos_theta_2(H);
        const Float H_wi = dr::dot(si.wi, H);
        //        const Float H_wo = dr::dot(wo, H);

        // GGX distribution
        Float root = ((Float) 1.f + Frame3f::tan_theta_2(H) / alpha2) * cos_theta2;
        const Float D = (Float) 1.f / (dr::Pi<Float> * alpha2 * root * root);

        const Float G = 4.f / ((1.f + dr::hypot((Float)1.f, alpha * Frame3f::tan_theta(si.wi))) *
                         (1.f + dr::hypot((Float)1.f, alpha * Frame3f::tan_theta(wo))));

        const UnpolarizedSpectrum F =
            m_fresnel(m_specular_reflectance->eval(si, active), H_wi);

        return (D * G / (4.f * cos_theta_i)) * F;
    }

    Float specular_pdf(Vector3f &H, Float cos_theta_i, const SurfaceInteraction3f &si, Mask active) const {
        Float roughness = dr::maximum(m_roughness->eval(si, active).x(), (Float) 1e-4);
        const Float alpha = roughness * roughness;
        const Float alpha2 = alpha * alpha;
        const Float cos_theta2 = Frame3f::cos_theta_2(H);

        Float root = ((Float) 1.f + Frame3f::tan_theta_2(H) / alpha2) * cos_theta2;
        const Float D = (Float)1.f / (dr::Pi<Float> * alpha2 * root * root);
        const Float G1 = 2.f / (1.f + dr::hypot((Float)1.f, alpha * Frame3f::tan_theta(si.wi)));

        return D * G1 / (4.f * cos_theta_i);
    }

    inline UnpolarizedSpectrum m_fresnel(const UnpolarizedSpectrum &F0,
                                         Float c) const {
        return F0 + (UnpolarizedSpectrum(1.0f) - F0) * dr::pow(1.f - c, 5.0f);
    }
};

MI_EXPORT_PLUGIN(CookTorrance);
NAMESPACE_END(mitsuba)

